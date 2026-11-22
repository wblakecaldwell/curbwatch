# nestmon

Check on your outdoor Nest cameras from the command line. `nestmon` pulls snapshots
through a [Starling Home Hub](https://www.starlinghome.io/)'s local Developer Connect
API and answers one practical question: **are the garbage cans out at the curb?**

Everything for the curb check runs on your own machine. A local vision model
(Qwen3-VL 30B-A3B via [LM Studio](https://lmstudio.ai/) by default, or any vision model
in [Ollama](https://ollama.com/)) is loaded only for the few seconds a check takes and
unloaded right after.

## How the curb check works

1. Grab a fresh snapshot from the configured camera.
2. Align it to saved "empty curb" reference frames using ORB features found
   *outside* the curb crop, so the cans coming and going don't disturb alignment.
   Small camera drift moves the crop along with the view; a large shift or a poor
   match is reported as "camera moved, re-mark the crop" instead of a guess.
3. Crop the curb region.
   - **Daytime:** ask the local vision model whether a trash can is present
     that isn't in the empty-curb reference.
   - **Night (IR):** the model can't see cans reliably in grayscale at a
     distance, so measure how much of the curb's background edges are hidden
     instead.
4. Print the result as JSON and keep the crop plus answer in `~/.local/share/nestmon/curb/history/`,
   so mistakes can be reviewed and thresholds calibrated.

## Requirements

- A Starling Home Hub with Developer Connect enabled, and an API key for it
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A local vision model, either:
  - [LM Studio](https://lmstudio.ai/) with its server running and a vision model downloaded
    (default: `qwen3-vl-30b-a3b-instruct-mlx`), or
  - [Ollama](https://ollama.com/) with a vision model (`ollama pull qwen2.5vl:7b`) and
    `LOCAL_BACKEND=ollama`
- Optional, for `snap --describe` and `ask`: an Anthropic API key

## Setup

```bash
git clone https://github.com/wblakecaldwell/nest-monitoring.git
cd nest-monitoring
uv sync
mkdir -p ~/.config/nestmon
cp .env.example ~/.config/nestmon/.env   # then set STARLING_HUB and STARLING_API_KEY
uv run nestmon check    # lists the hub's cameras
```

Config and pictures live outside the repo, so a clone never holds anything personal:

- Config: `~/.config/nestmon/.env` (or point `NESTMON_CONFIG` at another file)
- Curb setup, reference frames, history: `~/.local/share/nestmon/curb/`
- Snapshots: `~/.local/share/nestmon/snapshots/`

On macOS you can keep the keys in the Keychain instead; see `.env.example`.

## Usage

```bash
uv run nestmon check                      # hub status and cameras
uv run nestmon snap                       # one snapshot per camera
uv run nestmon snap --describe            # ...plus a short Claude description of each
uv run nestmon ask IMG... -q "question"   # ask Claude about saved images
```

## Set it up for your own curb

There's no model training involved. Qwen is used as-is; you "teach" the tool by
telling it where the cans go (a crop box) and showing it what that spot looks
like empty (reference frames). Then you check a few answers and tune one number
for night.

**1. Pick a camera and find the spot.** Choose the camera with the clearest view
of where your cans sit. Save a snapshot with a labeled pixel grid:

```bash
uv run nestmon grid --camera "front yard"    # any part of the camera's name works
```

Open `~/.local/share/nestmon/curb/grid_<camera>.jpg` and note a box around the cans' spot as
`x,y,width,height`. Make it a bit bigger than the cans, and include some
background behind them (the curb edge, a fence line). The night check relies on
the cans hiding that background.

**2. Mark the crop while the cans are NOT out.** Each reference frame doubles as
the "empty curb" picture that every check is compared against.

```bash
uv run nestmon cans mark --camera "front yard" --roi 600,380,420,200
```

Look at `~/.local/share/nestmon/curb/roi_preview.jpg` to confirm the red box is where you want it.
Re-run with a new `--roi` until it is.

**3. Add a reference for the other lighting.** Day and night (IR) frames don't
match each other, so you need one empty reference of each. If you marked in
daylight, wait until dark (cans still in) and run:

```bash
uv run nestmon cans mark --camera "front yard" --add-reference
```

Add more later if alignment fails at dawn or dusk, or after a season change.

**4. Check it against reality.** Run checks with the cans in and with them out,
in daylight and at night:

```bash
uv run nestmon cans check
```

Every answer is logged to `~/.local/share/nestmon/curb/history/log.jsonl`, with the crop saved next to
it. Daytime answers come from Qwen; if it's wrong, make the crop tighter around
the spot, or retake the daytime reference in similar light.

**5. Tune the night cutoff.** At night the result comes from
`background_visible.visible` (1.0 means the background is fully visible). Look at
that value in the log for night checks with cans out and with cans in, and set
`NIGHT_VISIBLE_CUTOFF` in your config between the two groups. If they overlap, the
cans are probably too far from the camera's IR light for night checks to work (set `NIGHT_CHECKS=off`);
schedule checks for daylight only.

**6. Run it on a schedule.** `cans check` prints JSON and exits with status 1
when it can't answer (camera moved, view blocked, no reference for the current
lighting). Call it from cron or launchd on the evening before pickup, and send
yourself a notification when `cans_out` is `false`.

If the camera gets bumped, checks report "camera may have moved". Run `grid`
and `cans mark` again to redo the crop.

## Configuration

All settings come from environment variables or `~/.config/nestmon/.env`. See
[`.env.example`](.env.example) for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `STARLING_HUB` | *(required)* | Hub IP or hostname |
| `STARLING_API_KEY` | *(required)* | Developer Connect API key |
| `STARLING_HTTPS_PORT` / `STARLING_HTTP_PORT` | `3443` / `3080` | Hub ports |
| `LOCAL_BACKEND` | `lmstudio` | `lmstudio` or `ollama` for the daytime check |
| `LMSTUDIO_URL` | `http://localhost:1234` | LM Studio server |
| `LM_API_TOKEN` | | LM Studio API token, if its server requires one |
| `LMSTUDIO_TTL` | `10` | Idle seconds before LM Studio unloads a JIT-loaded model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `LOCAL_VISION_MODEL` | `qwen3-vl-30b-a3b-instruct-mlx` (Ollama: `qwen2.5vl:7b`) | Model for the daytime check |
| `NIGHT_VISIBLE_CUTOFF` | `0.6` | Night: below this background visibility, cans are out |
| `NIGHT_CHECKS` | `on` | Set to `off` if night scores overlap; night checks then report "can't tell" |
| `MIN_INLIERS` | `25` | Minimum landmark matches to trust alignment |
| `MAX_SHIFT_FRAC` | `0.15` | Shift (fraction of width) treated as "camera moved" |
| `VISION_BACKEND` | `api` | `api` (Anthropic) or `agy` (Antigravity CLI) for `describe`/`ask` |
| `ANTHROPIC_API_KEY` | | Needed when `VISION_BACKEND=api` |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model for `describe`/`ask` |
| `NESTMON_CONFIG` | `~/.config/nestmon/.env` | Config file |
| `NESTMON_DATA_DIR` | `~/.local/share/nestmon/snapshots` | Where `snap` writes images |
| `NESTMON_STATE_DIR` | `~/.local/share/nestmon/curb` | Crop setup, references, history |

The night cutoff came from a handful of frames on one camera. Calibrate it for
yours from `~/.local/share/nestmon/curb/history/log.jsonl`.

## Privacy

Snapshots and curb data contain pictures of your property, and the config holds
your hub address and API key. All of it lives outside the repo by default, and
`.gitignore` also excludes `.env` files and images as a backstop. The hub uses a self-signed certificate,
so TLS verification is off for hub requests; that traffic stays on your LAN.
`snap --describe` and `ask` send images to the vision backend you choose. The
curb check never leaves your machine.

## License

[MIT](LICENSE)
