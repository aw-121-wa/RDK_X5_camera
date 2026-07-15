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
S: 80-255
V: 50-255
```

Adjust it if the lighting or ball color changes:

```bash
./build/blue_ball_detector --h-min 95 --h-max 125 --s-min 100 --v-min 60
```

Other useful options:

```bash
./build/blue_ball_detector --width 640 --height 480 --min-area 300 --rate-ms 100 --display
```

For higher processing/output frequency, remove the artificial delay:

```bash
./build/blue_ball_detector --rate-ms 0
```

## Tests

The test executable uses synthetic images to verify multi-ball detection, rightmost-ball selection, center offset, distance, overlay drawing, and CSV formatting:

```bash
cmake --build build --target blue_ball_detector_tests
ctest --test-dir build --output-on-failure
```
