#include "blue_ball_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/geometry/2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

struct ProgramOptions {
    std::string camera = "auto";
    int scan_max = 9;
    int width = 640;
    int height = 480;
    HSVRange hsv{90, 130, 80, 255, 50, 255};
    double min_area = 300.0;
    int rate_ms = 100;
    bool display = false;
};

constexpr const char* kDisplayWindowName = "Blue Ball Detector";

struct DetectionBuffers {
    cv::Mat hsv_frame;
    cv::Mat mask;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
};

struct CameraSelection {
    int index = -1;
    bool automatic = false;
    bool likely_external = false;
};

bool isFlag(const std::string& arg, const std::string& flag)
{
    return arg == flag || arg.rfind(flag + "=", 0) == 0;
}

std::string takeValue(int& index, int argc, char** argv, const std::string& arg, const std::string& flag)
{
    const std::string prefix = flag + "=";
    if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
    }

    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
    }

    ++index;
    return argv[index];
}

int parseIntValue(const std::string& value, const std::string& flag)
{
    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer for " + flag + ": " + value);
    }
    return parsed;
}

double parseDoubleValue(const std::string& value, const std::string& flag)
{
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid number for " + flag + ": " + value);
    }
    return parsed;
}

void printUsage(const char* program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "Options:\n"
        << "  --camera auto|N   Camera index or automatic scan. Default: auto\n"
        << "  --scan-max N      Maximum camera index to scan. Default: 9\n"
        << "  --width N         Capture width. Default: 640\n"
        << "  --height N        Capture height. Default: 480\n"
        << "  --h-min N         HSV hue minimum. Default: 90\n"
        << "  --h-max N         HSV hue maximum. Default: 130\n"
        << "  --s-min N         HSV saturation minimum. Default: 80\n"
        << "  --s-max N         HSV saturation maximum. Default: 255\n"
        << "  --v-min N         HSV value minimum. Default: 50\n"
        << "  --v-max N         HSV value maximum. Default: 255\n"
        << "  --min-area N      Minimum contour area. Default: 300\n"
        << "  --rate-ms N       Output interval in milliseconds. Default: 100\n"
        << "  --display         Show annotated camera preview for PC testing\n"
        << "  --help            Show this help message\n";
}

ProgramOptions parseOptions(int argc, char** argv)
{
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (isFlag(arg, "--camera")) {
            options.camera = takeValue(i, argc, argv, arg, "--camera");
        } else if (isFlag(arg, "--scan-max")) {
            options.scan_max = parseIntValue(takeValue(i, argc, argv, arg, "--scan-max"), "--scan-max");
        } else if (isFlag(arg, "--width")) {
            options.width = parseIntValue(takeValue(i, argc, argv, arg, "--width"), "--width");
        } else if (isFlag(arg, "--height")) {
            options.height = parseIntValue(takeValue(i, argc, argv, arg, "--height"), "--height");
        } else if (isFlag(arg, "--h-min")) {
            options.hsv.h_min = parseIntValue(takeValue(i, argc, argv, arg, "--h-min"), "--h-min");
        } else if (isFlag(arg, "--h-max")) {
            options.hsv.h_max = parseIntValue(takeValue(i, argc, argv, arg, "--h-max"), "--h-max");
        } else if (isFlag(arg, "--s-min")) {
            options.hsv.s_min = parseIntValue(takeValue(i, argc, argv, arg, "--s-min"), "--s-min");
        } else if (isFlag(arg, "--s-max")) {
            options.hsv.s_max = parseIntValue(takeValue(i, argc, argv, arg, "--s-max"), "--s-max");
        } else if (isFlag(arg, "--v-min")) {
            options.hsv.v_min = parseIntValue(takeValue(i, argc, argv, arg, "--v-min"), "--v-min");
        } else if (isFlag(arg, "--v-max")) {
            options.hsv.v_max = parseIntValue(takeValue(i, argc, argv, arg, "--v-max"), "--v-max");
        } else if (isFlag(arg, "--min-area")) {
            options.min_area = parseDoubleValue(takeValue(i, argc, argv, arg, "--min-area"), "--min-area");
        } else if (isFlag(arg, "--rate-ms")) {
            options.rate_ms = parseIntValue(takeValue(i, argc, argv, arg, "--rate-ms"), "--rate-ms");
        } else if (arg == "--display") {
            options.display = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (options.scan_max < 0) {
        throw std::invalid_argument("--scan-max must be >= 0");
    }
    if (options.width <= 0 || options.height <= 0) {
        throw std::invalid_argument("--width and --height must be > 0");
    }
    if (options.min_area < 0.0) {
        throw std::invalid_argument("--min-area must be >= 0");
    }
    if (options.rate_ms < 0) {
        throw std::invalid_argument("--rate-ms must be >= 0");
    }

    return options;
}

bool readValidFrame(cv::VideoCapture& capture)
{
    cv::Mat frame;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (capture.read(frame) && !frame.empty()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

bool containsUsbPathToken(const std::string& path)
{
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower.find("usb") != std::string::npos;
}

bool isLikelyExternalCameraIndex(int index)
{
#ifdef _WIN32
    return index > 0;
#else
    namespace fs = std::filesystem;
    const std::string video_name = "video" + std::to_string(index);
    const fs::path sysfs_device = fs::path("/sys/class/video4linux") / video_name / "device";

    std::error_code error;
    const fs::path canonical_device = fs::weakly_canonical(sysfs_device, error);
    if (!error && containsUsbPathToken(canonical_device.string())) {
        return true;
    }

    const fs::path by_path_dir("/dev/v4l/by-path");
    if (!fs::exists(by_path_dir, error) || error) {
        return false;
    }

    for (const fs::directory_entry& entry : fs::directory_iterator(by_path_dir, error)) {
        if (error) {
            break;
        }

        const fs::path entry_path = entry.path();
        if (!containsUsbPathToken(entry_path.string())) {
            continue;
        }

        std::error_code target_error;
        fs::path target = fs::read_symlink(entry_path, target_error);
        if (target_error) {
            continue;
        }

        if (target.is_relative()) {
            target = entry_path.parent_path() / target;
        }

        const fs::path canonical_target = fs::weakly_canonical(target, target_error);
        if (!target_error && canonical_target.filename() == video_name) {
            return true;
        }
    }

    return false;
#endif
}

CameraSelection resolveCameraSelection(const ProgramOptions& options)
{
    if (options.camera != "auto") {
        return CameraSelection{parseIntValue(options.camera, "--camera"), false, false};
    }

    const std::vector<CameraCandidate> candidates = scanCameraCandidates(options.scan_max);
    const int index = choosePreferredCameraIndex(candidates);
    bool likely_external = false;

    for (const CameraCandidate& candidate : candidates) {
        if (candidate.index == index) {
            likely_external = candidate.likely_external;
            break;
        }
    }

    return CameraSelection{index, true, likely_external};
}

DetectionResult selectRightmostDetection(const std::vector<DetectionResult>& detections)
{
    DetectionResult selected{false, -1, -1, 0.0};

    for (const DetectionResult& detection : detections) {
        if (!detection.found) {
            continue;
        }

        if (!selected.found || detection.x > selected.x ||
            (detection.x == selected.x && detection.area > selected.area)) {
            selected = detection;
        }
    }

    return selected;
}

std::vector<DetectionResult> detectBlueBallsWithBuffers(
    const cv::Mat& frame,
    const HSVRange& hsv,
    double min_area,
    DetectionBuffers& buffers)
{
    std::vector<DetectionResult> detections;

    if (frame.empty()) {
        return detections;
    }

    cv::cvtColor(frame, buffers.hsv_frame, cv::COLOR_BGR2HSV);

    cv::inRange(
        buffers.hsv_frame,
        cv::Scalar(hsv.h_min, hsv.s_min, hsv.v_min),
        cv::Scalar(hsv.h_max, hsv.s_max, hsv.v_max),
        buffers.mask);

    cv::morphologyEx(buffers.mask, buffers.mask, cv::MORPH_OPEN, buffers.kernel);
    cv::morphologyEx(buffers.mask, buffers.mask, cv::MORPH_CLOSE, buffers.kernel);

    const int component_count = cv::connectedComponentsWithStats(
        buffers.mask,
        buffers.labels,
        buffers.stats,
        buffers.centroids,
        8,
        CV_32S);

    detections.reserve(static_cast<std::size_t>(std::max(0, component_count - 1)));

    for (int label = 1; label < component_count; ++label) {
        const int area = buffers.stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area) {
            continue;
        }

        const double x = buffers.centroids.at<double>(label, 0);
        const double y = buffers.centroids.at<double>(label, 1);
        detections.push_back(DetectionResult{
            true,
            static_cast<int>(std::lround(x)),
            static_cast<int>(std::lround(y)),
            static_cast<double>(area)});
    }

    std::sort(
        detections.begin(),
        detections.end(),
        [](const DetectionResult& left, const DetectionResult& right) {
            if (left.x != right.x) {
                return left.x < right.x;
            }
            if (left.y != right.y) {
                return left.y < right.y;
            }
            return left.area > right.area;
        });

    return detections;
}

} // namespace

DetectionResult detectBlueBall(const cv::Mat& frame, const HSVRange& hsv, double min_area)
{
    return selectRightmostDetection(detectBlueBalls(frame, hsv, min_area));
}

std::vector<DetectionResult> detectBlueBalls(const cv::Mat& frame, const HSVRange& hsv, double min_area)
{
    DetectionBuffers buffers;
    return detectBlueBallsWithBuffers(frame, hsv, min_area, buffers);
}

FrameMetrics computeFrameMetrics(const cv::Size& frame_size, const DetectionResult& detection)
{
    const int center_x = frame_size.width / 2;
    const int center_y = frame_size.height / 2;

    if (!detection.found) {
        return FrameMetrics{false, -1, -1, center_x, center_y, 0, 0, -1.0};
    }

    const int dx = detection.x - center_x;
    const int dy = detection.y - center_y;
    const double distance = std::sqrt(static_cast<double>(dx) * dx + static_cast<double>(dy) * dy);

    return FrameMetrics{true, detection.x, detection.y, center_x, center_y, dx, dy, distance};
}

std::string formatMetricsCsv(const FrameMetrics& metrics)
{
    std::ostringstream output;
    output << "B,"
           << (metrics.found ? 1 : 0) << ','
           << metrics.ball_x << ','
           << metrics.ball_y << ','
           << metrics.center_x << ','
           << metrics.center_y << ','
           << metrics.dx << ','
           << metrics.dy << ','
           << std::fixed << std::setprecision(2) << metrics.distance;
    return output.str();
}

void drawOverlay(cv::Mat& frame, const FrameMetrics& metrics)
{
    const std::vector<DetectionResult> detections = metrics.found
        ? std::vector<DetectionResult>{DetectionResult{true, metrics.ball_x, metrics.ball_y, 0.0}}
        : std::vector<DetectionResult>{};
    drawOverlay(frame, metrics, detections);
}

void drawOverlay(cv::Mat& frame, const FrameMetrics& metrics, const std::vector<DetectionResult>& detections)
{
    if (frame.empty()) {
        return;
    }

    const cv::Point center(metrics.center_x, metrics.center_y);
    const cv::Scalar center_color(0, 255, 0);
    const cv::Scalar ball_color(0, 0, 255);
    const cv::Scalar other_ball_color(255, 255, 0);
    const cv::Scalar line_color(0, 255, 255);

    for (const DetectionResult& detection : detections) {
        if (!detection.found || (metrics.found && detection.x == metrics.ball_x && detection.y == metrics.ball_y)) {
            continue;
        }

        const cv::Point ball(detection.x, detection.y);
        cv::circle(frame, ball, 4, other_ball_color, cv::FILLED, cv::LINE_8);
        cv::drawMarker(frame, ball, other_ball_color, cv::MARKER_CROSS, 16, 1, cv::LINE_8);
    }

    if (metrics.found) {
        const cv::Point ball(metrics.ball_x, metrics.ball_y);
        cv::line(frame, center, ball, line_color, 2, cv::LINE_8);
    }

    cv::drawMarker(frame, center, center_color, cv::MARKER_CROSS, 24, 2, cv::LINE_8);
    cv::circle(frame, center, 4, center_color, cv::FILLED, cv::LINE_8);
    cv::putText(
        frame,
        "C(" + std::to_string(metrics.center_x) + "," + std::to_string(metrics.center_y) + ")",
        cv::Point(metrics.center_x + 8, metrics.center_y - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        center_color,
        1,
        cv::LINE_8);

    if (!metrics.found) {
        return;
    }

    const cv::Point ball(metrics.ball_x, metrics.ball_y);
    cv::drawMarker(frame, ball, ball_color, cv::MARKER_TILTED_CROSS, 24, 2, cv::LINE_8);
    cv::circle(frame, ball, 5, ball_color, cv::FILLED, cv::LINE_8);
    cv::putText(
        frame,
        "B(" + std::to_string(metrics.ball_x) + "," + std::to_string(metrics.ball_y) + ")",
        cv::Point(metrics.ball_x + 8, metrics.ball_y - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        ball_color,
        1,
        cv::LINE_8);
}

std::vector<CameraCandidate> scanCameraCandidates(int scan_max)
{
    std::vector<CameraCandidate> candidates;

    for (int index = 0; index <= scan_max; ++index) {
        cv::VideoCapture capture;
        if (!capture.open(index)) {
            continue;
        }

        if (readValidFrame(capture)) {
            candidates.push_back(CameraCandidate{index, true, isLikelyExternalCameraIndex(index)});
            capture.release();
        } else {
            capture.release();
        }
    }

    return candidates;
}

int choosePreferredCameraIndex(const std::vector<CameraCandidate>& candidates)
{
    int first_readable = -1;
    int first_external = -1;

    for (const CameraCandidate& candidate : candidates) {
        if (!candidate.readable) {
            continue;
        }

        if (first_readable < 0 || candidate.index < first_readable) {
            first_readable = candidate.index;
        }

        if (candidate.likely_external && (first_external < 0 || candidate.index < first_external)) {
            first_external = candidate.index;
        }
    }

    if (first_external >= 0) {
        return first_external;
    }

    return first_readable;
}

int findCameraIndex(int scan_max)
{
    return choosePreferredCameraIndex(scanCameraCandidates(scan_max));
}

int runBlueBallDetector(int argc, char** argv)
{
    ProgramOptions options;
    try {
        options = parseOptions(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        printUsage(argv[0]);
        return 2;
    }

    CameraSelection camera_selection;
    try {
        camera_selection = resolveCameraSelection(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        printUsage(argv[0]);
        return 2;
    }

    const int camera_index = camera_selection.index;

    if (camera_index < 0) {
        std::cerr << "Error: no available camera found in range 0-" << options.scan_max << "\n";
        return 1;
    }

    cv::VideoCapture capture;
    if (!capture.open(camera_index)) {
        std::cerr << "Error: failed to open camera index " << camera_index << "\n";
        return 1;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);

    std::cerr << "Using camera index " << camera_index;
    if (camera_selection.automatic) {
        std::cerr << " (auto";
        if (camera_selection.likely_external) {
            std::cerr << ", external preferred";
        } else {
            std::cerr << ", fallback";
        }
        std::cerr << ")";
    } else {
        std::cerr << " (manual)";
    }
    std::cerr << "\n";

    cv::Mat frame;
    DetectionBuffers detection_buffers;
    while (true) {
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "Warning: failed to read frame\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(options.rate_ms));
            continue;
        }

        const std::vector<DetectionResult> detections =
            detectBlueBallsWithBuffers(frame, options.hsv, options.min_area, detection_buffers);
        const DetectionResult detection = selectRightmostDetection(detections);
        const FrameMetrics metrics = computeFrameMetrics(frame.size(), detection);
        std::cout << formatMetricsCsv(metrics) << std::endl;

        if (options.display) {
            drawOverlay(frame, metrics, detections);
            cv::imshow(kDisplayWindowName, frame);

            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        if (options.rate_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.rate_ms));
        }
    }

    return 0;
}
