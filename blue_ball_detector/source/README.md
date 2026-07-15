# RDK X5 Blue Ball Detector

This project builds a small C++ OpenCV program for RDK X5. It automatically finds a usable USB camera, detects a blue ball, and prints fixed CSV lines that are easy for a lower controller to parse.

## Build on RDK X5

Install the C++ build tools and OpenCV development package first. Package names can vary by image, but a typical Debian/Ubuntu-based image uses:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

Build:

```bash
cmake -S . -B build
cmake --build build
```

Run:

```bash
./build/blue_ball_detector
```

## PC Display Test

By default the program only prints CSV lines and does not open a window. This is the safest mode for RDK X5 or lower-controller communication.

For PC camera testing, enable the annotated preview window:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto --display
```

The preview marks the frame center as `C(x,y)`, the detected blue ball center as `B(x,y)`, and draws a line between them. Press `q` or `Esc` in the preview window to exit.

If multiple blue balls are visible, the preview marks all detected blue balls. The rightmost blue ball is highlighted and is the one used for CSV output.

To also mark the warehouse cell that contains each blue ball, enable grid detection and provide the fixed grid size:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto --display --grid-enable --grid-rows 3 --grid-cols 4 --cell-aspect 1.333333
```

The preview first looks for the warehouse's stable visual markers: the continuous black top line and the separated white short lines below each cell. Those marker lines are used to derive the yellow cell boxes and labels from left to right, top to bottom. If the marker-based method cannot find a valid grid, the program falls back to the older full-grid-line detector. If grid detection fails for a frame, the blue-ball overlay and CSV output continue normally and no wrong cell box is drawn.

Grid overlay uses the last valid grid for a short time to avoid flickering when one or two grid lines are missed in a frame. The default cache is 15 frames:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto --display --grid-enable --grid-rows 3 --grid-cols 4 --grid-cache-frames 15
```

If false yellow boxes are still common, lower the allowed aspect tolerance. If the grid is not detected because the camera is slightly tilted or the perspective is stronger, raise it a little:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera 1 --display --grid-enable --grid-rows 3 --grid-cols 4 --cell-aspect 1.333333 --grid-aspect-tolerance 0.35
```

Typical tuning range:

- `--grid-aspect-tolerance 0.25`: stricter, fewer false boxes.
- `--grid-aspect-tolerance 0.45`: looser, easier to detect under perspective or imperfect mounting.

For faster PC startup, use the known camera index instead of automatic scanning:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera 1 --display --grid-enable --grid-rows 3 --grid-cols 4 --cell-aspect 1.333333
```

If automatic scanning is needed, limiting the scan range also reduces startup time:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto --scan-max 2 --display --grid-enable --grid-rows 3 --grid-cols 4
```

Run without `--display` for pure CSV output:

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto
```

## Output Protocol

Each frame prints one CSV line:

```text
B,found,ball_x,ball_y,center_x,center_y,dx,dy,distance
```

Example when the blue ball is found:

```text
B,1,320,210,320,240,0,-30,30.00
```

Example when the blue ball is not found:

```text
B,0,-1,-1,320,240,0,0,-1.00
```

Fields:

- `found`: `1` means found, `0` means not found.
- `ball_x`, `ball_y`: selected blue ball center in pixels. If multiple blue balls are found, this is the rightmost blue ball.
- `center_x`, `center_y`: frame center in pixels.
- `dx`, `dy`: `ball - center`.
- `distance`: pixel distance from ball center to frame center.

## Camera Selection

Default behavior scans camera indexes `0` through `9` and prefers an external USB camera when one is detected:

```bash
./build/blue_ball_detector
```

On RDK X5/Linux, automatic selection checks video device paths such as `/sys/class/video4linux` and `/dev/v4l/by-path` for USB cameras. On Windows PC testing, automatic selection prefers a readable camera index greater than `0`, because built-in cameras are commonly index `0` and external USB cameras are commonly index `1` or higher. If no external candidate is found, the program falls back to the lowest readable camera index.

Startup logs show the selected camera index and whether automatic selection used an external-preferred candidate or fallback.

Manually choose a camera:

```bash
./build/blue_ball_detector --camera 2
```

Change scan range:

```bash
./build/blue_ball_detector --camera auto --scan-max 15
```

## Common Tuning

The default HSV range is suitable for many blue objects:

```text
H: 90-130
S: 60-255
V: 40-255
```

Adjust it if the lighting or ball color changes:

```bash
./build/blue_ball_detector --h-min 95 --h-max 125 --s-min 100 --v-min 60
```

Other useful options:

```bash
./build/blue_ball_detector --width 640 --height 480 --min-area 150 --rate-ms 100 --display
```

Warehouse grid overlay options:

```bash
./build/blue_ball_detector --display --grid-enable --grid-rows 3 --grid-cols 4 --grid-cache-frames 15 --cell-aspect 1.333333 --grid-aspect-tolerance 0.35
```

`--grid-enable` only affects the preview window. The CSV protocol remains the same 9 fields.
`--cell-aspect` is the physical width/height ratio of one warehouse cell. The default is `1.333333` (4:3).
`--grid-aspect-tolerance` is the accepted relative error of the fallback whole-grid aspect ratio. The default is `0.35`.

For higher processing/output frequency, remove the artificial delay:

```bash
./build/blue_ball_detector --rate-ms 0
```

## Tests

The test executable uses synthetic images to verify multi-ball detection, rightmost-ball selection, center offset, distance, overlay drawing, warehouse grid cell numbering, and CSV formatting:

```bash
cmake --build build --target blue_ball_detector_tests
ctest --test-dir build --output-on-failure
```
