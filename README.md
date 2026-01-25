# Ominis

Ominis is a block-stacking game built with Pygame. It lets you play with pentominoes, tetrominoes, and smaller ominoes, or train/watch an AI agent that learns to place pieces.

## Features
- Play mode with configurable piece sets and size ranges (1-5 blocks).
- AI training with adjustable architecture, exploration schedule, and optional headless mode.
- Watch mode for saved models, with optional MP4 recording.
- MIDI soundtrack and sound effects.

## Requirements
- Python 3
- Dependencies listed in `requirements.txt`

Notes:
- AI modes require PyTorch. The game can still run without it, but training/watch will be disabled.
- Recording uses PyAV and optional Windows-only loopback audio via PyAudioWPatch.

## Install
```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Build (Windows, PyInstaller)
This produces a console-enabled build with full functionality (audio, video recording, AI).

Option A: run the script
```bash
.\build.ps1
```

Option B: run the command directly
```bash
pyinstaller --noconfirm --clean --name Ominis --onedir --console \
  --add-data "assets;assets" \
  --add-data "settings.json;." \
  --add-data "recording.json;." \
  --collect-all pygame \
  --collect-binaries av \
  main.py
```

Output: `dist/Ominis/Ominis.exe`

## Controls
- Arrow keys: move left/right and soft drop
- Comma/period: rotate
- Space: hard drop
- F1: pause
- Esc: return to menu

## AI training and models
- Start training from the Train AI menu.
- Models are stored as `models/model-{hidden_size}-{hidden_layers}.pth`.
- Progress logs are written to `progress/model-{hidden_size}-{hidden_layers}.csv`.
- Training auto-saves periodically; you can also save custom names from the UI.

## AI training settings (Train AI menu)
- Hidden Size: size of each hidden layer in the network (16-2048).
- Hidden Layers: number of hidden layers in the network (1-4).
- Rand. (start -> floor): epsilon-greedy exploration range; starts at the first value and decays toward the floor.
- L.Rate (start -> end): learning rate decay range; starts at the first value and decays toward the end value.
- R&L Half-life: number of training batches for both epsilon and learning rate to halve.
- Piece Size: min -> max block count for pieces used during training (1-5).
- Big Piece Weight: biases sampling toward larger pieces (higher = more big pieces).
- Piece History: how many recent pieces share the reward signal; also sets Short Games length.
- Short Games: restart episodes after the Piece History count.
- View Training: render the board during training (off = headless/faster).
- Volume: training audio level (mainly noticeable when View Training is on).
- Holes (Decrease / Increase): reward/penalty per hole removed/added after placement.
- Jaggedness (Decrease / Increase): reward/penalty per decrease/increase in surface roughness.
- Pits (Decrease / Increase): reward/penalty per valley removed/added after placement.
- Max Height (Increase): penalty per increase in tallest column height.
- Height St.dev (Decrease / Increase): reward/penalty per change in column height variation.
- Game Over: penalty applied when the game ends.
- Lines Cleared: base reward per cleared line.
- Scale line reward per lines squared: if enabled, line reward scales with lines_cleared^2 instead of linear.

## Recording
- Recording is available in Watch AI mode only.
- Clips are saved to `Video/` as MP4 files.
- Audio capture is supported on Windows using WASAPI loopback; other platforms record video only.

## License
MIT. See `license.txt`.
