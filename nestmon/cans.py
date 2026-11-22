"""Are the garbage cans out at the curb?

Pipeline per check:
  1. Pull a fresh snapshot from the configured camera.
  2. Align it to saved reference frame(s) using ORB features found OUTSIDE the
     crop region (the cans themselves come and go, so they are masked out).
     Small camera shifts move the crop along; a poor match means the camera
     moved and the crop needs re-marking.
  3. Crop the curb region, upscale, and ask a local vision model (LM Studio by
     default, or Ollama). The model is unloaded from memory right after it answers.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import httpx
import numpy as np

from . import config

STATE_DIR = config.STATE_DIR
SETUP_FILE = STATE_DIR / "cans.json"
import shutil
import subprocess

# lmstudio (default) or ollama
LOCAL_BACKEND = (config.get("LOCAL_BACKEND", "lmstudio") or "lmstudio").lower()
LMSTUDIO_URL = config.get("LMSTUDIO_URL", "http://localhost:1234")
OLLAMA = config.get("OLLAMA_URL", "http://localhost:11434")
LOCAL_MODEL = config.get(
    "LOCAL_VISION_MODEL",
    "qwen3-vl-30b-a3b-instruct-mlx" if LOCAL_BACKEND == "lmstudio" else "qwen2.5vl:7b",
)

# Night (IR) cutoff for the background-visible score. In the original test, cans out
# scored 0.32-0.44 across 9 night frames and the empty curb scored 0.72. Your camera,
# curb and cans differ, so calibrate this from your own history/log.jsonl.
NIGHT_VISIBLE_CUTOFF = config.get_float("NIGHT_VISIBLE_CUTOFF", 0.6)
# Set NIGHT_CHECKS=off when night scores for cans in/out overlap (curb beyond the IR light).
NIGHT_CHECKS = (config.get("NIGHT_CHECKS", "on") or "on").lower() not in ("off", "0", "false", "no")

MIN_INLIERS = config.get_int("MIN_INLIERS", 25)              # below this, don't trust the alignment
MAX_SHIFT_FRAC = config.get_float("MAX_SHIFT_FRAC", 0.15)    # larger shift (fraction of width) = "camera moved"

# Wording matters. Telling qwen2.5vl "this is where the cans go", or forcing a JSON
# schema with a "cans_out" key, made it report cans on an empty curb every time.
# A neutral yes/no question against an empty-curb reference answered correctly.
PROMPT = (
    "Image 1 is a reference photo of an outdoor area with nothing on it. Image 2 is the same "
    "view now. Is there a trash can or garbage bin in image 2 that is not in image 1? Ignore "
    "lighting and noise differences. Answer yes or no first, then explain briefly."
)


@dataclass
class Setup:
    camera_id: str
    camera_name: str
    roi: tuple[int, int, int, int]          # x, y, w, h in reference-frame pixels
    references: list[str]                    # paths (relative to STATE_DIR) of reference frames

    @classmethod
    def load(cls) -> "Setup":
        if not SETUP_FILE.exists():
            raise SystemExit("No crop configured yet. Run: nestmon cans mark --camera NAME --roi x,y,w,h")
        d = json.loads(SETUP_FILE.read_text())
        return cls(d["camera_id"], d["camera_name"], tuple(d["roi"]), d["references"])

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        SETUP_FILE.write_text(json.dumps(self.__dict__, indent=2) + "\n")


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _mask_outside(shape, roi, pad: float = 0.5) -> np.ndarray:
    """Mask that excludes the ROI (plus padding) so moving cans don't affect alignment."""
    h, w = shape[:2]
    x, y, rw, rh = roi
    px, py = int(rw * pad), int(rh * pad)
    m = np.full((h, w), 255, np.uint8)
    m[max(0, y - py):min(h, y + rh + py), max(0, x - px):min(w, x + rw + px)] = 0
    m[: int(h * 0.08), int(w * 0.85):] = 0   # Nest watermark in the corner
    return m


@dataclass
class Alignment:
    ok: bool
    inliers: int
    dx: float
    dy: float
    reference: str
    note: str = ""


def is_night(img: np.ndarray) -> bool:
    """Nest night vision is grayscale: the color channels are (nearly) identical."""
    b, g, r = cv2.split(img.astype(np.int16))
    return float(np.abs(b - g).mean() + np.abs(g - r).mean()) < 3.0


def align(frame: np.ndarray, setup: Setup) -> tuple[Alignment, np.ndarray | None]:
    """Find the transform from reference -> current frame. Tries each reference, keeps the best."""
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    g = _gray(frame)
    kp2, des2 = orb.detectAndCompute(g, _mask_outside(g.shape, setup.roi))
    best: tuple[Alignment, np.ndarray | None] = (Alignment(False, 0, 0, 0, "", "no features"), None)
    if des2 is None:
        return best
    night = is_night(frame)
    for ref_name in setup.references:
        ref = cv2.imread(str(STATE_DIR / ref_name))
        if is_night(ref) != night:   # day and night frames don't match reliably
            continue
        rg = _gray(ref)
        kp1, des1 = orb.detectAndCompute(rg, _mask_outside(rg.shape, setup.roi))
        if des1 is None:
            continue
        pairs = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is None:
            continue
        n_in = int(inl.sum())
        if n_in > best[0].inliers:
            best = (Alignment(True, n_in, float(M[0, 2]), float(M[1, 2]), ref_name), M)
    a, M = best
    w = frame.shape[1]
    if not any(is_night(cv2.imread(str(STATE_DIR / r))) == night for r in setup.references):
        a.ok, a.note = False, f"no {'night' if night else 'daytime'} empty-curb reference yet; run: nestmon cans mark --camera {setup.camera_name!r} --add-reference (with the cans in)"
        return a, M
    if a.inliers < MIN_INLIERS:
        a.ok, a.note = False, f"only {a.inliers} landmark matches; camera may have moved or view is obscured"
        return a, M
    # Report how far the curb crop itself moved (not M's raw translation, which a
    # small rotation about the image corner inflates), plus rotation and zoom.
    x, y, rw, rh = setup.roi
    cx, cy = cv2.transform(np.float32([[[x + rw / 2, y + rh / 2]]]), M)[0, 0]
    a.dx, a.dy = float(cx - (x + rw / 2)), float(cy - (y + rh / 2))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    a.note = f"corrected shift ({a.dx:.0f}, {a.dy:.0f}) px, rotation {angle:.1f} deg, zoom {scale:.2f}"
    if abs(a.dx) > w * MAX_SHIFT_FRAC or abs(a.dy) > w * MAX_SHIFT_FRAC:
        a.ok, a.note = False, f"view shifted by ({a.dx:.0f}, {a.dy:.0f}) px; re-mark the crop"
    elif abs(angle) > 8 or not 0.9 < scale < 1.1:
        a.ok, a.note = False, f"view rotated {angle:.1f} deg / zoom {scale:.2f}; re-mark the crop"
    return a, M


def crop(frame: np.ndarray, setup: Setup, M: np.ndarray | None, out_size: int = 768) -> np.ndarray:
    x, y, w, h = setup.roi
    H, W = frame.shape[:2]
    if M is not None:
        # M maps reference -> current. Warp the current frame back into the reference's
        # coordinates (undoing shift, small rotation and zoom) so the fixed crop lines up
        # pixel-for-pixel with the empty-curb reference crop.
        frame = cv2.warpAffine(frame, cv2.invertAffineTransform(M), (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x, y = max(0, x), max(0, y)
    c = frame[y:min(H, y + h), x:min(W, x + w)]
    scale = out_size / max(c.shape[:2])
    return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else c


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()


def change_score(ref_crop: np.ndarray, cur_crop: np.ndarray) -> float:
    """Fraction of the crop that differs noticeably from the empty reference (0..1).
    Informational: lighting changes (day vs night) also raise it."""
    a = cv2.GaussianBlur(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = cv2.absdiff(a, b)
    return float((diff > 35).mean())


def occlusion(ref_crop: np.ndarray, cur_crop: np.ndarray, band: float = 0.5) -> dict:
    """How much of the far-side background (curb edge, posts) is still visible.

    Cans standing at the curb hide the strong horizontal edges behind them. We take the
    strongest edges in the top part of the empty reference crop and measure how much of
    that edge strength survives in the current crop, overall and per column strip.
    1.0 = fully visible; low values in some strips = something is in the way.
    First data point (2026-11-20 night): empty ~0.93 (all strips >= 0.84),
    cans out ~0.64 with strips down to 0.25. Thresholds still need calibrating."""
    def edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = g[: int(g.shape[0] * band)]
        g = cv2.GaussianBlur(cv2.medianBlur(g, 5).astype(np.float32), (5, 5), 0)
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5))
    cur_crop = cv2.resize(cur_crop, (ref_crop.shape[1], ref_crop.shape[0]))
    er, ec = edges(ref_crop), edges(cur_crop)
    mask = er > np.percentile(er, 95)
    strips = []
    step = max(1, er.shape[1] // 10)
    for x in range(0, er.shape[1] - step + 1, step):
        m = mask[:, x:x + step]
        if m.sum() > 20:
            strips.append(round(float(ec[:, x:x + step][m].mean() / er[:, x:x + step][m].mean()), 2))
    return {"visible": round(float(ec[mask].mean() / er[mask].mean()), 2),
            "min_strip": min(strips) if strips else None, "strips": strips}


def _parse(text: str, model: str) -> dict:
    text = text.strip()
    first = text.split()[0].strip(".,!:*").lower() if text else ""
    return {
        "cans_out": True if first == "yes" else False if first == "no" else None,
        "reason": text,
        "model": model,
    }


def _ask_ollama(ref_crop: np.ndarray, cur_crop: np.ndarray, model: str) -> dict:
    body = {
        "model": model,
        "prompt": PROMPT,
        "images": [_b64(ref_crop), _b64(cur_crop)],
        "stream": False,
        "keep_alive": 0,          # unload from memory right after answering
        "options": {"temperature": 0},
    }
    r = httpx.post(f"{OLLAMA}/api/generate", json=body, timeout=300)
    r.raise_for_status()
    return _parse(r.json()["response"], model)


def _lms_loaded(client: httpx.Client, model: str) -> bool:
    """Whether LM Studio already has the model in memory (LM Studio's native REST API)."""
    try:
        r = client.get(f"{LMSTUDIO_URL}/api/v0/models/{model}")
        return r.status_code == 200 and r.json().get("state") == "loaded"
    except (httpx.HTTPError, ValueError):
        return False


def _lms_unload(model: str) -> None:
    lms = config.get("LMS_BIN") or shutil.which("lms") or str(Path.home() / ".lmstudio/bin/lms")
    if Path(lms).exists():
        subprocess.run([lms, "unload", model], capture_output=True, timeout=60)


def _ask_lmstudio(ref_crop: np.ndarray, cur_crop: np.ndarray, model: str) -> dict:
    token = config.get("LM_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    img = lambda c: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(c)}"}}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT}, img(ref_crop), img(cur_crop)]}],
        "temperature": 0,
        "max_tokens": 300,
        # JIT-loaded models unload after this many idle seconds (backstop for the unload below).
        "ttl": config.get_int("LMSTUDIO_TTL", 10),
    }
    with httpx.Client(headers=headers, timeout=600) as client:
        was_loaded = _lms_loaded(client, model)
        try:
            r = client.post(f"{LMSTUDIO_URL}/v1/chat/completions", json=body)
            if r.status_code == 401:
                raise SystemExit("LM Studio needs an API token: set LM_API_TOKEN in "
                                 f"{config.CONFIG_FILE} or the Keychain (service nestmon-lmstudio)")
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
        finally:
            if not was_loaded:   # leave it loaded only if someone else had loaded it
                _lms_unload(model)
    return _parse(text, model)


def ask_local(ref_crop: np.ndarray, cur_crop: np.ndarray, model: str | None = None) -> dict:
    model = model or LOCAL_MODEL
    if LOCAL_BACKEND == "ollama":
        return _ask_ollama(ref_crop, cur_crop, model)
    return _ask_lmstudio(ref_crop, cur_crop, model)


def check(snapshot_jpeg: bytes, setup: Setup) -> dict:
    frame = decode(snapshot_jpeg)
    a, M = align(frame, setup)
    result: dict = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera": setup.camera_name,
        "alignment": a.__dict__,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(STATE_DIR / "latest_frame.jpg"), frame)
    if not a.ok:
        result["cans_out"] = None
        result["error"] = a.note
    else:
        c = crop(frame, setup, M)
        ref = crop(cv2.imread(str(STATE_DIR / a.reference)), setup, None)
        cv2.imwrite(str(STATE_DIR / "latest_crop.jpg"), c)
        result["change_score"] = round(change_score(ref, c), 3)
        occ = occlusion(ref, c)
        result["background_visible"] = occ
        if is_night(frame) and not NIGHT_CHECKS:
            # NIGHT_CHECKS=off: when the curb is beyond the IR light, an empty curb can score
            # the same as cans out (seen in testing: 0.34 for both), so report "can't tell"
            # instead of guessing.
            result["method"] = "night: disabled"
            result["cans_out"] = None
            result["error"] = "night-vision frame: night checks are off (curb too far for the camera to see cans in the dark)"
        elif is_night(frame):
            # The local model can't see cans in night IR; use background occlusion.
            result["method"] = "night: background visibility"
            result["cans_out"] = occ["visible"] < NIGHT_VISIBLE_CUTOFF
        else:
            result["method"] = "day: local vision model"
            result.update(ask_local(ref, c))
    co = result.get("cans_out")
    result["status"] = "out" if co is True else "in" if co is False else "unknown"
    (STATE_DIR / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
    # Keep every crop with its answer, so mistakes can be corrected later and,
    # if ever needed, used to train a small classifier.
    if "error" not in result:
        hist = STATE_DIR / "history"
        hist.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        result["crop"] = f"history/{stamp}.jpg"
        cv2.imwrite(str(hist / f"{stamp}.jpg"), c)
        with open(hist / "log.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


def mark(snapshot_jpeg: bytes, camera_id: str, camera_name: str,
         roi: tuple[int, int, int, int] | None, add_reference: bool) -> Setup:
    """Save a reference frame and (re)define the crop region.

    Take references only when the cans are NOT at the curb: each one doubles as the
    "empty curb" picture the model compares against. Add a daytime and a night one."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    frame = decode(snapshot_jpeg)
    name = f"ref_{datetime.now():%Y%m%dT%H%M%S}.jpg"
    cv2.imwrite(str(STATE_DIR / name), frame)
    if add_reference and SETUP_FILE.exists():
        s = Setup.load()
        s.references.append(name)
    else:
        if roi is None:
            raise SystemExit("--roi x,y,w,h is required when defining a new crop")
        s = Setup(camera_id, camera_name, roi, [name])
    s.save()
    # Preview with the crop outlined, for a human to sanity-check.
    x, y, w, h = s.roi
    prev = frame.copy()
    cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imwrite(str(STATE_DIR / "roi_preview.jpg"), prev)
    return s
