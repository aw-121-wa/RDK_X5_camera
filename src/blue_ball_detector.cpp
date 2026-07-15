#include "blue_ball_detector.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

int resolveCameraIndex(const ProgramOptions& options)
{
    if (options.camera == "auto") {
        return findCameraIndex(options.scan_max);
    }

    return parseIntValue(options.camera, "--camera");
}

} // namespace

DetectionResult detectBlueBall(const cv::Mat& frame, const HSVRange& hsv, double min_area)
{
    if (frame.empty()) {
        return DetectionResult{false, -1, -1, 0.0};
    }

    cv::Mat hsv_frame;
    cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(
        hsv_frame,
        cv::Scalar(hsv.h_min, hsv.s_min, hsv.v_min),
        cv::Scalar(hsv.h_max, hsv.s_max, hsv.v_max),
        mask);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_area = 0.0;
    int best_index = -1;

    for (std::size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area >= min_area && area > best_area) {
            best_area = area;
            best_index = static_cast<int>(i);
        }
    }

    if (best_index < 0) {
        return DetectionResult{false, -1, -1, 0.0};
    }

    const cv::Moments moments = cv::moments(contours[best_index]);
    if (std::abs(moments.m00) <= std::numeric_limits<double>::epsilon()) {
        return DetectionResult{false, -1, -1, 0.0};
    }

    const int x = static_cast<int>(std::lround(moments.m10 / moments.m00));
    const int y = static_cast<int>(std::lround(moments.m01 / moments.m00));

    return DetectionResult{true, x, y, best_area};
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

int findCameraIndex(int scan_max)
{
    for (int index = 0; index <= scan_max; ++index) {
        cv::VideoCapture capture;
        if (!capture.open(index)) {
            continue;
        }

        if (readValidFrame(capture)) {
            capture.release();
            return index;
        }
    }

    return -1;
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

    int camera_index = -1;
    try {
        camera_index = resolveCameraIndex(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        printUsage(argv[0]);
        return 2;
    }

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

    std::cerr << "Using camera index " << camera_index << "\n";

    cv::Mat frame;
    while (true) {
        if (!capture.read(frame) || frame.empty()) {
            std::cerr << "Warning: failed to read frame\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(options.rate_ms));
            continue;
        }

        const DetectionResult detection = detectBlueBall(frame, options.hsv, options.min_area);
        const FrameMetrics metrics = computeFrameMetrics(frame.size(), detection);
        std::cout << formatMetricsCsv(metrics) << std::endl;

        if (options.rate_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(options.rate_ms));
        }
    }

    return 0;
}
