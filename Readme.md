# 🌈 Hand Wavelength

**Control the color of light with your hands.**

Hand Wavelength is a real-time computer-vision app that turns the distance between your two hands into a wavelength of visible light. Move your hands together and the scene glows violet; pull them apart and it shifts through blue, green and yellow to red. Bring your hands all the way together and every color combines into white light.

<!-- Add a demo GIF or screenshot here, e.g.: ![demo](assets/demo.gif) -->

## ✨ Features

- **Distance → wavelength:** hand separation maps linearly onto the visible spectrum (380–700 nm)
- **Physically inspired color:** wavelength is converted to RGB with the Dan Bruton approximation
- **Live HUD:** wavelength (nm), distance %, hands detected, FPS, gesture and spread
- **Glowing visuals:** neon hand skeletons, light beams between fingertips, and particle trails
- **White light mode:** touch your hands together and all colors blend into white
- **Spectrum bar:** a live marker shows where you are on the 380–700 nm range
- **Works with one hand too:** thumb-to-pinky spread drives the color
- **Smooth & fast:** exponential smoothing removes jitter; runs in real time on a normal laptop webcam

## 🔬 How it works

| Hands | Wavelength | Color |
|-------|-----------|-------|
| Close together | ~380 nm | Violet |
| ↓ | ~450 nm | Blue |
| ↓ | ~530 nm | Green |
| ↓ | ~590 nm | Yellow / Orange |
| Far apart | ~700 nm | Red |
| Touching | all wavelengths | White |

1. **Hand tracking:** [MediaPipe](https://developers.google.com/mediapipe) detects up to 2 hands and returns 21 landmarks per hand.
2. **Distance:** the app finds the center of each palm and measures the gap between them as a fraction of the frame width.
3. **Mapping:** that distance is normalized to a 0–1 value and smoothed, then mapped to `380 + t × 320` nm.
4. **Color:** the wavelength is converted to an RGB color and used for the skeleton, beams, particles and glow (Gaussian-blurred layers added on top of the video).
5. **White light:** when the palms get closer than a threshold, the screen eases to white, since white light is the combination of all visible wavelengths.

## 📦 Requirements

- Python **3.10 – 3.12**
- A webcam
- Internet access on first run (newer MediaPipe versions download a ~8 MB hand model once)

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/wavelength-hands.git
cd wavelength-hands

# optional but recommended
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Usage

```bash
python wavelength_hands.py
```

Hold both hands up to the camera and move them closer together or further apart.

### Controls

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit |
| `h` | Show / hide the HUD |
| `f` | Toggle fullscreen |

## ⚙️ Configuration

Edit the constants at the top of `wavelength_hands.py`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `CAM_INDEX` | `0` | Which camera to use |
| `FRAME_W`, `FRAME_H` | `1280`, `720` | Capture resolution |
| `DIST_NEAR` | `0.06` | Hand distance (fraction of frame width) that gives violet |
| `DIST_FAR` | `0.70` | Hand distance that gives red |
| `TOUCH_THRESHOLD` | `0.07` | Distance below which white light mode starts |
| `SMOOTHING` | `0.25` | 0–1; lower is smoother, higher is more responsive |

## 🛠️ Troubleshooting

- **Camera doesn't open:** change `CAM_INDEX` to `1` or `2`, and make sure no other app is using the camera.
- **`pip install mediapipe` fails:** use Python 3.10–3.12 (MediaPipe doesn't always support the newest Python versions).
- **Model download fails:** download `hand_landmarker.task` manually from the [MediaPipe hand landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and place it next to `wavelength_hands.py`.
- **Hands not detected:** use good lighting and keep your hands fully inside the frame.
- **Low FPS:** lower `FRAME_W` / `FRAME_H`, or close other camera-heavy apps.

## 📁 Project structure

```
wavelength-hands/
├── wavelength_hands.py   # the whole app
├── requirements.txt
└── README.md
```

## 🧰 Built with

- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/): camera capture and rendering
- [MediaPipe](https://developers.google.com/mediapipe): hand landmark detection
- [NumPy](https://numpy.org/)

## 💡 Ideas for extending it

- Sound: map wavelength to pitch
- Record the output to a video file
- A browser version with MediaPipe JS
- Pinch gestures to change brightness or particle density

## 📄 License

This project is released under the MIT License. Add a `LICENSE` file to your repository to make it official.

## 🙌 Credits

Inspired by a viral "distance = wavelength" hand-tracking demo. Wavelength-to-RGB conversion based on Dan Bruton's approximation.