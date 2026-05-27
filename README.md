# Assignment 4: Computer Vision and Markers

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Perspective Transformation

```bash
python3 perspective_transformation/image_extractor.py perspective_transformation/sample_image.jpg output.jpg --width 300 --height 200
```

Click four points to select a region. Press `ESC` to reset and `S` to save the warped result.

## AR Game

```bash
python3 ar_game/AR_game.py
```

Use `python3 ar_game/AR_game.py 1` if the wrong webcam opens.

Place `DICT_6X6_250` ArUco markers at the corners of a sheet of paper. Each detected marker becomes a colored button. Colored coins fall from the top and keep falling if tracking briefly drops. Move or rotate the sheet so the matching marker button catches the coin. Correct hits add score, wrong hits or missed coins cost lives.
