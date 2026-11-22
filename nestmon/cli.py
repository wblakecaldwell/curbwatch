"""nestmon: pull Nest snapshots via a Starling hub and ask Claude about them.

  nestmon check                 hub status + list of cameras
  nestmon snap [--describe]     save one snapshot per camera (optionally describe each)
  nestmon ask IMAGE... -q TEXT  ask Claude a question about saved images
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

from . import config
from .starling import Starling

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "camera"


def cmd_check(args) -> None:
    hub = Starling(https=not args.http)
    print("Hub status:", json.dumps(hub.status(), indent=2))
    devs = hub.devices()
    print(f"\n{len(devs)} devices:")
    for d in devs:
        print(f"  [{d.get('category') or d.get('type')}] {d.get('name')}  id={d.get('id')}")
    if args.raw:
        print(json.dumps(devs, indent=2))


def cmd_snap(args) -> None:
    hub = Starling(https=not args.http)
    stamp = datetime.now().astimezone()
    saved: list[Path] = []
    for cam in hub.cameras():
        name = cam.get("name") or cam["id"]
        try:
            img = hub.snapshot(cam["id"], width=args.width)
        except Exception as e:  # keep going if one camera is offline
            print(f"  {name}: snapshot failed: {e}")
            continue
        out = config.DATA_DIR / slug(name) / stamp.strftime("%Y-%m-%d") / f"{stamp:%Y%m%dT%H%M%S%z}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img)
        saved.append(out)
        print(f"  {name}: {out} ({len(img)//1024} KB)")
        if args.describe:
            from .vision import describe
            text = describe(out)
            out.with_suffix(".txt").write_text(text + "\n")
            print(f"    {text}\n")


def cmd_ask(args) -> None:
    from .vision import ask
    print(ask([Path(p) for p in args.images], args.question))


def _find_camera(hub, name: str) -> dict:
    cams = hub.cameras()
    for c in cams:
        if name.lower() in (c.get("name") or "").lower() or name == c.get("id"):
            return c
    raise SystemExit(f"No camera matching {name!r}. Cameras: {[c.get('name') for c in cams]}")


def cmd_cans_mark(args) -> None:
    from . import cans
    hub = Starling(https=not args.http)
    cam = _find_camera(hub, args.camera)
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    s = cans.mark(hub.snapshot(cam["id"]), cam["id"], cam["name"], roi, args.add_reference)
    print(f"Saved. Camera={s.camera_name} roi={s.roi} references={len(s.references)}")
    print(f"Preview: {cans.STATE_DIR / 'roi_preview.jpg'}")


def cmd_cans_check(args) -> None:
    from . import cans
    s = cans.Setup.load()
    hub = Starling(https=not args.http)
    r = cans.check(hub.snapshot(s.camera_id), s)
    print(json.dumps(r, indent=2))
    if r.get("cans_out") is None:
        raise SystemExit(1)


def cmd_grid(args) -> None:
    """Save a snapshot with a labeled grid so a human can point at a region."""
    import cv2
    from . import cans
    hub = Starling(https=not args.http)
    cam = _find_camera(hub, args.camera)
    img = cans.decode(hub.snapshot(cam["id"]))
    h, w = img.shape[:2]
    step = args.step
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out = cans.STATE_DIR / f"grid_{slug(cam['name'])}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(out)


def main() -> None:
    p = argparse.ArgumentParser(prog="nestmon")
    p.add_argument("--http", action="store_true", help="use the hub's plain-HTTP port (default 3080) instead of HTTPS (3443)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("--raw", action="store_true")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("snap")
    s.add_argument("--describe", action="store_true")
    s.add_argument("--width", type=int, default=None)
    s.set_defaults(fn=cmd_snap)

    a = sub.add_parser("ask")
    a.add_argument("images", nargs="+")
    a.add_argument("-q", "--question", required=True)
    a.set_defaults(fn=cmd_ask)

    g = sub.add_parser("grid", help="snapshot with pixel grid, for choosing a crop")
    g.add_argument("--camera", required=True)
    g.add_argument("--step", type=int, default=120)
    g.set_defaults(fn=cmd_grid)

    cn = sub.add_parser("cans", help="garbage-can-at-the-curb check")
    csub = cn.add_subparsers(dest="cans_cmd", required=True)
    m = csub.add_parser("mark", help="save a reference frame and the crop region")
    m.add_argument("--camera", required=True)
    m.add_argument("--roi", help="x,y,w,h in pixels of the full snapshot")
    m.add_argument("--add-reference", action="store_true",
                   help="keep the crop, just add another reference frame (e.g. daytime)")
    m.set_defaults(fn=cmd_cans_mark)
    ck = csub.add_parser("check", help="are the cans out right now?")
    ck.set_defaults(fn=cmd_cans_check)

    args = p.parse_args()
    from .starling import StarlingError
    try:
        args.fn(args)
    except StarlingError as e:
        raise SystemExit(f"Starling hub error: {e}")


if __name__ == "__main__":
    main()
