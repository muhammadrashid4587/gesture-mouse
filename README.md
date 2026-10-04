# gesture-mouse

Control an on-screen board with both hands through your webcam. The window shows the
live camera with both hand skeletons drawn (left hand orange, right hand blue) and a board
you work with directly: reach for something on screen and pinch it where your fingers are.

- **Buttons:** pinch on one to press it. It changes color and counts the presses.
- **Cards:** pinch a card, move your hand to drag it, and open your fingers to drop it.
  Open your fingers *while swiping* to throw it: it slides on, slows down and bounces off the edges.
  Pinch a sliding card to catch it.
- **Slider:** pinch the knob and move it.

Each hand works on its own, so you can drag both cards at once.

| Gesture | What it does |
| --- | --- |
| Move your hand | The ring between your thumb and index finger is that hand's cursor; it follows your fingers, smoothed |
| Pinch (thumb + index) | Press / grab what is under the ring; it moves with your hand; open to drop |
| Swipe and let go | Throws the card in that direction |
| Closed fist | Freezes that hand only: it lets go and ignores the board; open to resume |
| Hand leaves view | Anything it held is dropped where it was (never thrown) |

## Run

```sh
uv run python gesture_mouse.py
```

Press **q** (or Esc) to quit, **f** for fullscreen, and **d** to show each hand's live pinch numbers.
Use `--camera 1` to pick another camera.

The first run downloads the MediaPipe hand model (`hand_landmarker.task`, about 8 MB) next to the script.

## Camera permission (macOS)

The first time you run it, macOS asks to let your terminal app (Terminal, iTerm, VS Code, ...)
use the camera. Click **Allow**. If you denied it before, or the window never shows the camera:

1. Open **System Settings > Privacy & Security > Camera**.
2. Turn on your terminal app.
3. Quit and reopen the terminal, then run the command again.

## Tuning

The gesture thresholds at the top of `gesture_mouse.py` were fitted to a recording of real pinches,
quick clicks and fists. If clicks are missed or fire on their own, press **d** and watch the
`pinch` number: it should drop below `PINCH_ON` (0.22) when your fingertips touch and rise above
`PINCH_OFF` (0.30) when you let go. While you hold something it has to rise above `HOLD_OFF` (0.42),
so a pinch that loosens a little as you move does not drop it. Cursor smoothing is
`POINTER_SMOOTHING` / `POINTER_RESPONSE`; throwing is `THROW_MIN` (slowest swipe that throws) and
`FRICTION` (higher = cards stop sooner).

## Notes

- Python is pinned to 3.12, and `mediapipe` is pinned to 1.0.0 because 1.0.1 aborts on Apple Silicon
  with `Check failed: service_ Service is unavailable`.
- This only moves things inside the window. It does not control the real Mac mouse.
