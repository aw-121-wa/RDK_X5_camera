# RDK X5 蓝球检测器

本项目是一个为 RDK X5 构建的轻量级 C++ / OpenCV 程序。它能自动发现可用的 USB 摄像头，检测蓝色球体，并以固定格式的 CSV 行输出结果，方便下位机控制器解析。

## 在 RDK X5 上编译

首先安装 C++ 构建工具和 OpenCV 开发包。包名可能因镜像而异，典型的 Debian/Ubuntu 镜像使用以下命令：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

编译：

```bash
cmake -S . -B build
cmake --build build
```

运行：

```bash
./build/blue_ball_detector
```

## PC 端显示测试

默认情况下，程序仅输出 CSV 行，不会打开任何窗口。这是适合 RDK X5 或下位机通信的最安全模式。

若要在 PC 上进行摄像头测试，可以启用带标注的预览窗口：

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto --display
```

预览窗口中会标记画面中心 `C(x,y)`、检测到的蓝球中心 `B(x,y)`，并在两者之间绘制连线。在预览窗口中按 `q` 或 `Esc` 键退出。

如果画面中有多个蓝球，预览会标记所有检测到的蓝球。其中最右侧的蓝球会被高亮显示，并作为 CSV 输出的目标。

不带 `--display` 运行即为纯 CSV 输出模式：

```powershell
D:\RDK_X5\blue_ball_detector\bin\blue_ball_detector.exe --camera auto
```

## 输出协议

每帧输出一行 CSV：

```text
B,found,ball_x,ball_y,center_x,center_y,dx,dy,distance
```

检测到蓝球时的示例：

```text
B,1,320,210,320,240,0,-30,30.00
```

未检测到蓝球时的示例：

```text
B,0,-1,-1,320,240,0,0,-1.00
```

字段说明：

- `found`：`1` 表示检测到，`0` 表示未检测到。
- `ball_x`、`ball_y`：选中的蓝球中心像素坐标。当检测到多个蓝球时，取最右侧的蓝球。
- `center_x`、`center_y`：画面中心的像素坐标。
- `dx`、`dy`：`ball - center`（蓝球相对于画面中心的偏移量）。
- `distance`：蓝球中心到画面中心的像素距离。

## 摄像头选择

默认行为会扫描摄像头索引 `0` 到 `9`，并在检测到外部 USB 摄像头时优先选择：

```bash
./build/blue_ball_detector
```

在 RDK X5 / Linux 上，自动选择会检查 `/sys/class/video4linux` 和 `/dev/v4l/by-path` 等视频设备路径以识别 USB 摄像头。在 Windows PC 测试时，自动选择会优先选择索引大于 `0` 的可读摄像头，因为内置摄像头通常为索引 `0`，而外部 USB 摄像头通常为索引 `1` 或更高。如果没有找到外部摄像头候选，程序会回退到最低可读的摄像头索引。

启动日志会显示所选摄像头的索引，以及自动选择是使用了外部优先候选还是进行了回退。

手动指定摄像头：

```bash
./build/blue_ball_detector --camera 2
```

修改扫描范围：

```bash
./build/blue_ball_detector --camera auto --scan-max 15
```

## 常用调参

默认 HSV 范围适用于大多数蓝色物体：

```text
H: 90-130
S: 80-255
V: 50-255
```

如果光照条件或球体颜色发生变化，可以调整参数：

```bash
./build/blue_ball_detector --h-min 95 --h-max 125 --s-min 100 --v-min 60
```

其他常用选项：

```bash
./build/blue_ball_detector --width 640 --height 480 --min-area 300 --rate-ms 100 --display
```

如需更高的处理/输出频率，可以移除人工延迟：

```bash
./build/blue_ball_detector --rate-ms 0
```

## 测试

测试程序使用合成图像来验证多球检测、最右侧球选择、中心偏移量、距离计算、叠加绘制以及 CSV 格式化：

```bash
cmake --build build --target blue_ball_detector_tests
ctest --test-dir build --output-on-failure
```
