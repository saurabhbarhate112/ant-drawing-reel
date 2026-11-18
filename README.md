# 4 Ways to Draw a Butterfly / an Ant (programmatic pencil reels)

A 1080x1920 drawing reel rendered entirely in code. One ant is drawn four ways (Simple, Normal, Better, Realistic) on toothy grey paper by a real hand cut-out holding a pencil.

- `reel.py`: ant geometry, a stroke timeline with real drawing speeds, the hand motion (wrist lag and swivel, lifts between strokes, motion blur, soft shadow), the shot plan (hard cuts, close framing so only the pencil and fingertips show), coloured-pencil-on-paper ink, and a synthesised pencil-on-paper sound.
- `assets/`: the hand cut-out (graphite, red and brown pencil versions) and the label overlay.
- `tools/`: scripts that made the assets.

Render: the `render` GitHub Actions workflow renders 6 chunks in parallel, joins them, adds the sound and uploads `ant-reel` as an artifact.

Local checks: `python reel.py --plan` (shots and hand-edge check), `python reel.py --still 3,12,30` (PNG frames to `out/`).

## Butterfly reel (current)

`butterfly.py` draws one monarch four times in the same pose with rising skill (Simple, Normal, Better, Realistic) under a locked-off overhead camera like the reference. The hand is the horizontal right-edge pose (`assets/hand2_*`). Strokes are generated to move like a person's: short overlapping pulls, light sketch then firmer lines, 2/3-power-law speed inside a minimum-jerk envelope, colour in small back-and-forth patches, and fingers that move the pencil while the hand lags. The `render` workflow builds `butterfly-reel`. `reel.py` is the earlier ant version.
