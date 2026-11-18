# 4 Ways to Draw an Ant (programmatic pencil reel)

A 1080x1920 drawing reel rendered entirely in code. One ant is drawn four ways (Simple, Normal, Better, Realistic) on toothy grey paper by a real hand cut-out holding a pencil.

- `reel.py`: ant geometry, a stroke timeline with real drawing speeds, the hand motion (wrist lag and swivel, lifts between strokes, motion blur, soft shadow), the shot plan (hard cuts, close framing so only the pencil and fingertips show), coloured-pencil-on-paper ink, and a synthesised pencil-on-paper sound.
- `assets/`: the hand cut-out (graphite, red and brown pencil versions) and the label overlay.
- `tools/`: scripts that made the assets.

Render: the `render` GitHub Actions workflow renders 6 chunks in parallel, joins them, adds the sound and uploads `ant-reel` as an artifact.

Local checks: `python reel.py --plan` (shots and hand-edge check), `python reel.py --still 3,12,30` (PNG frames to `out/`).
