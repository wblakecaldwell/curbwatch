"""Ask Claude about camera snapshots."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import anthropic

from . import config

SYSTEM = (
    "You review still frames from a homeowner's outdoor security cameras. "
    "Describe only what is visible. If something is ambiguous (darkness, glare, "
    "occlusion, low resolution), say so plainly instead of guessing."
)

DESCRIBE = (
    "In 2-4 sentences, summarize what this camera shows right now: the scene, "
    "lighting/weather, any people, vehicles, packages, or animals, and anything "
    "that looks out of the ordinary."
)


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.require("ANTHROPIC_API_KEY"))


AGY_SCHEMA = json.dumps({
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
})


def _ask_agy(images: list[Path], question: str, model: str | None) -> str:
    """Route through the Antigravity CLI (agy), billed to your Antigravity plan."""
    agy = config.AGY_BIN or shutil.which("agy") or str(Path.home() / ".local/bin/agy")
    paths = "\n".join(f"- {p.resolve()}" for p in images)
    prompt = (
        f"{SYSTEM}\n\nOpen and look at these image files:\n{paths}\n\n{question}\n\n"
        "Put only your answer about the images in the `answer` field."
    )
    dirs = sorted({str(p.resolve().parent) for p in images})
    cmd = [agy, "-p", prompt, "--model", model or config.AGY_MODEL, "--mode", "plan",
           "--output-format", "json", "--json-schema", AGY_SCHEMA, "--print-timeout", "180s"]
    for d in dirs:
        cmd += ["--add-dir", d]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=dirs[0], timeout=240)
    if r.returncode != 0:
        raise RuntimeError(f"agy failed ({r.returncode}): {r.stderr.strip()[:500]}")
    out = json.loads(r.stdout)
    if out.get("status") != "SUCCESS":
        raise RuntimeError(f"agy status {out.get('status')}")
    return (out.get("structured_output") or {}).get("answer") or out.get("response", "").strip()


def ask(images: list[Path], question: str, model: str | None = None) -> str:
    if config.VISION_BACKEND == "agy":
        return _ask_agy(images, question, model)
    content: list[dict] = []
    for p in images:
        content.append({"type": "text", "text": f"Image: {p.name}"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png",
                "data": base64.standard_b64encode(p.read_bytes()).decode(),
            },
        })
    content.append({"type": "text", "text": question})

    resp = _client().messages.create(
        model=model or config.MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        return "(Claude declined to answer this one.)"
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def describe(image: Path, model: str | None = None) -> str:
    return ask([image], DESCRIBE, model)
