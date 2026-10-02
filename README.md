# Squat Counter

![tests](https://github.com/abroroff/squat-counter/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time desktop app that counts squats through your webcam — and **only counts the ones done properly**.
Built with MediaPipe Pose (3D landmarks), OpenCV and PySide6.

Most rep counters just watch one knee angle, so they can be fooled by lifting a leg, half-squatting or shuffling around.
This one validates every rep against a set of biomechanical rules and tells you *why* a rep was rejected.

## Features

- **Live pose tracking** — skeleton overlay; legs turn yellow → green at full depth, red on a bad rep
- **Strict rep validation** — a rep counts only if *all* of these hold:
  1. The whole body (shoulders, hips, knees, ankles) is visible
  2. You first stand still for ~1 s so the app calibrates your hip and foot positions
  3. **Both** knees bend past the target angle (measured in 3D, not 2D)
  4. The hips actually drop (≥ 25 % of leg length)
  5. Both knees bend symmetrically — lifting one leg doesn't work
  6. Feet stay planted — no heel lifts, no stepping
  7. The bottom position is held briefly and you return to full standing
- **Instant feedback** — "Go lower…", "Good! Now stand up", or the exact reject reason
- **Adjustable depth** — slider for the required knee angle (70°–120°)
- **Counted / rejected stats**, depth progress bar, live knee angles
- **Camera-independent core logic** with a unit test suite that simulates squats frame by frame

## How it works

```
Webcam ─► OpenCV ─► MediaPipe PoseLandmarker ─► SquatCounter (state machine) ─► PySide6 UI
                     (image + world landmarks)    NO_PERSON → CALIBRATING
                                                  → READY ⇄ SQUATTING
```

| File | Role |
|---|---|
| [`main.py`](main.py) | Camera thread, pose detection, skeleton drawing, Qt UI |
| [`squat_logic.py`](squat_logic.py) | Pure-Python state machine — calibration, depth, symmetry and foot checks |
| [`tests/test_squat_logic.py`](tests/test_squat_logic.py) | Synthetic-pose tests for valid reps and every cheat case |

Knee angles come from MediaPipe's **world landmarks** (metric 3D), so they stay accurate even when you're not exactly side-on to the camera.
Hip drop and foot movement are normalised by your calibrated leg length, so distance from the camera doesn't matter.

## Getting started

Requires Python 3.12 and a webcam.

```bash
git clone https://github.com/abroroff/squat-counter.git
cd squat-counter
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py            # or: python main.py --camera 1
```

The pose model (`pose_landmarker_full.task`, ~9 MB) is downloaded automatically on first run.

> **macOS:** grant camera access to your Terminal / IDE in *System Settings → Privacy & Security → Camera*.

### Usage tips

- Stand 2–3 m from the camera with your whole body in frame
- Stand still for a second until the status says **Ready**
- Squat — the counter beeps on every valid rep

## Tests

```bash
pytest -v
```

The tests feed synthetic pose sequences into `SquatCounter` and cover: valid reps, missing calibration, shallow squats, tiny bobs, one-leg lifts, asymmetric knees, heel lifts, too-fast reps, leaving the frame, and moving feet.

## Tech stack

Python · MediaPipe Tasks (Pose Landmarker) · OpenCV · NumPy · PySide6 (Qt 6) · pytest · GitHub Actions

---

## O'zbekcha

Kamera orqali o'tirib-turishlarni (squat) real vaqtda sanaydigan dastur. Faqat **to'g'ri bajarilgan** takrorlar sanaladi:
ikkala tizza yetarlicha bukilishi, tos pastga tushishi, oyoqlar yerdan uzilmasligi va joyidan siljimasligi shart.
Noto'g'ri takrorda dastur sababini ko'rsatadi ("yetarlicha pastga tushmadingiz", "oyoq yerdan ko'tarildi" va h.k.).

Ishga tushirish: yuqoridagi *Getting started* bo'limiga qarang.

## License

[MIT](LICENSE)
