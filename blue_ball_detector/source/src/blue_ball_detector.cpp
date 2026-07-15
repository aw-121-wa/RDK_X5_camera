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
#include <numeric>
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
    HSVRange hsv{90, 130, 60, 255, 40, 255};
    double min_area = 150.0;
    int rate_ms = 100;
    bool display = false;
    GridConfig grid{false, 0, 0};
    int grid_cache_frames = 15;
};

constexpr const char* kDisplayWindowName = "Blue Ball Detector";

struct DetectionBuffers {
    cv::Mat hsv_frame;
    cv::Mat mask;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
};

struct CameraSelection {
    int index = -1;
    bool automatic = false;
    bool likely_external = false;
};

struct GridBoundsCandidate {
    bool found = false;
    int left = 0;
    int right = 0;
    int top = 0;
    int bottom = 0;
    double ratio_error = std::numeric_limits<double>::infinity();
    int support_count = 0;
    double area = 0.0;
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
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid integer for " + flag + ": " + value);
    }
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
        << "  --s-min N         HSV saturation minimum. Default: 60\n"
        << "  --s-max N         HSV saturation maximum. Default: 255\n"
        << "  --v-min N         HSV value minimum. Default: 40\n"
        << "  --v-max N         HSV value maximum. Default: 255\n"
        << "  --min-area N      Minimum contour area. Default: 150\n"
        << "  --rate-ms N       Output interval in milliseconds. Default: 100\n"
        << "  --display         Show annotated camera preview for PC testing\n"
        << "  --grid-enable     Detect warehouse grid and draw blue-ball cells in display mode\n"
        << "  --grid-rows N     Warehouse grid row count. Required with --grid-enable\n"
        << "  --grid-cols N     Warehouse grid column count. Required with --grid-enable\n"
        << "  --grid-cache-frames N  Reuse last valid grid for N missed frames. Default: 15\n"
        << "  --cell-aspect N   Physical cell width/height ratio. Default: 1.333333\n"
        << "  --grid-aspect-tolerance N  Accepted grid aspect error ratio. Default: 0.35\n"
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
        } else if (arg == "--grid-enable") {
            options.grid.enabled = true;
        } else if (isFlag(arg, "--grid-rows")) {
            options.grid.rows = parseIntValue(takeValue(i, argc, argv, arg, "--grid-rows"), "--grid-rows");
        } else if (isFlag(arg, "--grid-cols")) {
            options.grid.cols = parseIntValue(takeValue(i, argc, argv, arg, "--grid-cols"), "--grid-cols");
        } else if (isFlag(arg, "--grid-cache-frames")) {
            options.grid_cache_frames =
                parseIntValue(takeValue(i, argc, argv, arg, "--grid-cache-frames"), "--grid-cache-frames");
        } else if (isFlag(arg, "--cell-aspect")) {
            options.grid.cell_aspect = parseDoubleValue(takeValue(i, argc, argv, arg, "--cell-aspect"), "--cell-aspect");
        } else if (isFlag(arg, "--grid-aspect-tolerance")) {
            options.grid.aspect_tolerance = parseDoubleValue(
                takeValue(i, argc, argv, arg, "--grid-aspect-tolerance"),
                "--grid-aspect-tolerance");
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
    if (options.grid.enabled && (options.grid.rows <= 0 || options.grid.cols <= 0)) {
        throw std::invalid_argument("--grid-rows and --grid-cols must be > 0 when --grid-enable is set");
    }
    if (!options.grid.enabled && (options.grid.rows < 0 || options.grid.cols < 0)) {
        throw std::invalid_argument("--grid-rows and --grid-cols must be >= 0");
    }
    if (options.grid_cache_frames < 0) {
        throw std::invalid_argument("--grid-cache-frames must be >= 0");
    }
    if (options.grid.cell_aspect <= 0.0) {
        throw std::invalid_argument("--cell-aspect must be > 0");
    }
    if (options.grid.aspect_tolerance <= 0.0) {
        throw std::invalid_argument("--grid-aspect-tolerance must be > 0");
    }

    return options;
}

bool readValidFrame(cv::VideoCapture& capture)
{
    cv::Mat frame;
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (capture.read(frame) && !frame.empty()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
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

std::vector<int> clusterLinePositions(std::vector<int> positions, int tolerance)
{
    std::vector<int> clustered;
    if (positions.empty()) {
        return clustered;
    }

    std::sort(positions.begin(), positions.end());

    int sum = positions.front();
    int count = 1;
    int current_center = positions.front();

    for (std::size_t index = 1; index < positions.size(); ++index) {
        const int position = positions[index];
        if (std::abs(position - current_center) <= tolerance) {
            sum += position;
            ++count;
            current_center = static_cast<int>(std::lround(static_cast<double>(sum) / count));
        } else {
            clustered.push_back(current_center);
            sum = position;
            count = 1;
            current_center = position;
        }
    }

    clustered.push_back(current_center);
    return clustered;
}

std::vector<int> chooseGridLinePositions(const std::vector<int>& clustered, int expected_count)
{
    if (expected_count <= 1 || static_cast<int>(clustered.size()) < expected_count) {
        return {};
    }

    if (static_cast<int>(clustered.size()) == expected_count) {
        return clustered;
    }

    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(expected_count));

    const int first = clustered.front();
    const int last = clustered.back();
    for (int index = 0; index < expected_count; ++index) {
        const double expected = first + (last - first) * (static_cast<double>(index) / (expected_count - 1));
        const auto closest = std::min_element(
            clustered.begin(),
            clustered.end(),
            [expected](int left, int right) {
                return std::abs(left - expected) < std::abs(right - expected);
            });
        selected.push_back(*closest);
    }

    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    if (static_cast<int>(selected.size()) != expected_count) {
        return {};
    }

    return selected;
}

GridDetectionResult buildGridCells(const std::vector<int>& x_lines, const std::vector<int>& y_lines, const GridConfig& config)
{
    GridDetectionResult result{false, {}};

    if (static_cast<int>(x_lines.size()) != config.cols + 1 ||
        static_cast<int>(y_lines.size()) != config.rows + 1) {
        return result;
    }

    result.found = true;
    result.cells.reserve(static_cast<std::size_t>(config.rows * config.cols));

    for (int row = 0; row < config.rows; ++row) {
        for (int col = 0; col < config.cols; ++col) {
            const float left = static_cast<float>(x_lines[col]);
            const float right = static_cast<float>(x_lines[col + 1]);
            const float top = static_cast<float>(y_lines[row]);
            const float bottom = static_cast<float>(y_lines[row + 1]);

            result.cells.push_back(GridCell{
                row,
                col,
                row * config.cols + col + 1,
                std::vector<cv::Point2f>{
                    cv::Point2f(left, top),
                    cv::Point2f(right, top),
                    cv::Point2f(right, bottom),
                    cv::Point2f(left, bottom)}});
        }
    }

    return result;
}

std::vector<int> normalizedLinePositions(std::vector<int> lines)
{
    std::sort(lines.begin(), lines.end());
    lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
    return lines;
}

double expectedGridAspect(const GridConfig& config)
{
    if (config.rows <= 0 || config.cols <= 0 || config.cell_aspect <= 0.0) {
        return 0.0;
    }

    return (static_cast<double>(config.cols) * config.cell_aspect) / static_cast<double>(config.rows);
}

double gridAspectError(int left, int right, int top, int bottom, const GridConfig& config)
{
    const int width = right - left;
    const int height = bottom - top;
    const double expected_ratio = expectedGridAspect(config);

    if (width <= 0 || height <= 0 || expected_ratio <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double actual_ratio = static_cast<double>(width) / static_cast<double>(height);
    return std::abs((actual_ratio / expected_ratio) - 1.0);
}

int countLinesInBounds(const std::vector<int>& lines, int first, int last)
{
    return static_cast<int>(std::count_if(
        lines.begin(),
        lines.end(),
        [first, last](int line) {
            return line >= first && line <= last;
        }));
}

bool isBetterGridBoundsCandidate(const GridBoundsCandidate& candidate, const GridBoundsCandidate& best)
{
    if (!best.found) {
        return true;
    }

    constexpr double kEpsilon = 1e-9;
    if (candidate.ratio_error < best.ratio_error - kEpsilon) {
        return true;
    }
    if (candidate.ratio_error > best.ratio_error + kEpsilon) {
        return false;
    }
    if (candidate.support_count != best.support_count) {
        return candidate.support_count > best.support_count;
    }
    return candidate.area > best.area;
}

GridBoundsCandidate chooseBestGridBoundsCandidate(
    const std::vector<int>& x_lines,
    const std::vector<int>& y_lines,
    const GridConfig& config)
{
    const std::vector<int> normalized_x = normalizedLinePositions(x_lines);
    const std::vector<int> normalized_y = normalizedLinePositions(y_lines);
    GridBoundsCandidate best;

    if (normalized_x.size() < 2 || normalized_y.size() < 2 || expectedGridAspect(config) <= 0.0) {
        return best;
    }

    for (std::size_t left_index = 0; left_index + 1 < normalized_x.size(); ++left_index) {
        for (std::size_t right_index = left_index + 1; right_index < normalized_x.size(); ++right_index) {
            const int left = normalized_x[left_index];
            const int right = normalized_x[right_index];

            for (std::size_t top_index = 0; top_index + 1 < normalized_y.size(); ++top_index) {
                for (std::size_t bottom_index = top_index + 1; bottom_index < normalized_y.size(); ++bottom_index) {
                    const int top = normalized_y[top_index];
                    const int bottom = normalized_y[bottom_index];
                    const double ratio_error = gridAspectError(left, right, top, bottom, config);

                    if (ratio_error > config.aspect_tolerance) {
                        continue;
                    }

                    const double area = static_cast<double>(right - left) * static_cast<double>(bottom - top);
                    if (area <= 0.0) {
                        continue;
                    }

                    const GridBoundsCandidate candidate{
                        true,
                        left,
                        right,
                        top,
                        bottom,
                        ratio_error,
                        countLinesInBounds(normalized_x, left, right) +
                            countLinesInBounds(normalized_y, top, bottom),
                        area};

                    if (isBetterGridBoundsCandidate(candidate, best)) {
                        best = candidate;
                    }
                }
            }
        }
    }

    return best;
}

std::vector<int> fillEvenlySpacedLinesFromBounds(int first, int last, int expected_count)
{
    if (expected_count <= 1 || first == last) {
        return {};
    }

    if (first > last) {
        std::swap(first, last);
    }

    std::vector<int> filled;
    filled.reserve(static_cast<std::size_t>(expected_count));
    for (int index = 0; index < expected_count; ++index) {
        const double ratio = static_cast<double>(index) / (expected_count - 1);
        filled.push_back(static_cast<int>(std::lround(first + (last - first) * ratio)));
    }

    return filled;
}

bool gridAspectMatches(const std::vector<int>& x_lines, const std::vector<int>& y_lines, const GridConfig& config)
{
    if (static_cast<int>(x_lines.size()) != config.cols + 1 ||
        static_cast<int>(y_lines.size()) != config.rows + 1 ||
        x_lines.empty() ||
        y_lines.empty()) {
        return false;
    }

    const double ratio_error = gridAspectError(x_lines.front(), x_lines.back(), y_lines.front(), y_lines.back(), config);
    return ratio_error <= config.aspect_tolerance;
}

struct MarkerSegment {
    int left = 0;
    int right = 0;
    int center_x = 0;
    int center_y = 0;
    int width = 0;
    int height = 0;
    int area = 0;
};

struct MarkerRow {
    int center_y = 0;
    std::vector<MarkerSegment> segments;
};

struct MarkerColumnSelection {
    bool found = false;
    std::vector<MarkerSegment> segments;
    double spacing_variance = std::numeric_limits<double>::max();
    int span = 0;
};

int clampInt(int value, int low, int high)
{
    return std::max(low, std::min(value, high));
}

int detectTopBlackLineY(const cv::Mat& frame)
{
    if (frame.empty()) {
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat dark_mask;
    cv::threshold(gray, dark_mask, 35, 255, cv::THRESH_BINARY_INV);
    if (dark_mask.rows > 1) {
        dark_mask.rowRange(dark_mask.rows / 2, dark_mask.rows).setTo(0);
    }

    const int kernel_width = std::max(25, frame.cols / 5);
    cv::Mat closed;
    cv::morphologyEx(
        dark_mask,
        closed,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_width, 3)));

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(closed, labels, stats, centroids, 8, CV_32S);

    const int min_width = std::max(60, frame.cols / 3);
    const int max_height = std::max(8, frame.rows / 12);

    int best_label = -1;
    int best_width = 0;
    int best_area = 0;
    for (int label = 1; label < count; ++label) {
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        if (top >= frame.rows / 2 || width < min_width || height > max_height) {
            continue;
        }

        if (width > best_width || (width == best_width && area > best_area)) {
            best_label = label;
            best_width = width;
            best_area = area;
        }
    }

    if (best_label < 0) {
        return -1;
    }

    const int top = stats.at<int>(best_label, cv::CC_STAT_TOP);
    const int height = stats.at<int>(best_label, cv::CC_STAT_HEIGHT);
    return top + height / 2;
}

std::vector<MarkerSegment> detectWhiteMarkerSegments(const cv::Mat& frame, int top_y, const GridConfig& config)
{
    std::vector<MarkerSegment> segments;
    if (frame.empty() || config.cols <= 0) {
        return segments;
    }

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 170), cv::Scalar(179, 90, 255), white_mask);
    if (top_y > 0) {
        const int clear_bottom = clampInt(top_y + 4, 0, white_mask.rows);
        white_mask.rowRange(0, clear_bottom).setTo(0);
    }

    cv::morphologyEx(
        white_mask,
        white_mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3)));
    cv::morphologyEx(
        white_mask,
        white_mask,
        cv::MORPH_OPEN,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 2)));

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(white_mask, labels, stats, centroids, 8, CV_32S);

    const int min_width = std::max(10, frame.cols / std::max(12, config.cols * 8));
    const int max_width =
        std::max(min_width + 1, static_cast<int>((frame.cols / static_cast<double>(config.cols)) * 1.25));
    const int max_height = std::max(8, frame.rows / 12);

    for (int label = 1; label < count; ++label) {
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);

        if (width < min_width || width > max_width || height < 1 || height > max_height) {
            continue;
        }

        const double aspect = width / static_cast<double>(std::max(1, height));
        if (aspect < 3.0 || area < std::max(12, width * height / 5)) {
            continue;
        }

        MarkerSegment segment;
        segment.left = left;
        segment.right = left + width;
        segment.center_x = static_cast<int>(std::lround(centroids.at<double>(label, 0)));
        segment.center_y = static_cast<int>(std::lround(centroids.at<double>(label, 1)));
        segment.width = width;
        segment.height = height;
        segment.area = area;
        segments.push_back(segment);
    }

    return segments;
}

std::vector<MarkerRow> groupMarkerRows(std::vector<MarkerSegment> segments, int min_segments, int frame_rows)
{
    std::vector<MarkerRow> rows;
    if (segments.empty()) {
        return rows;
    }

    std::sort(segments.begin(), segments.end(), [](const MarkerSegment& a, const MarkerSegment& b) {
        if (a.center_y == b.center_y) {
            return a.center_x < b.center_x;
        }
        return a.center_y < b.center_y;
    });

    const int y_tolerance = std::max(5, frame_rows / 60);
    MarkerRow current;
    int y_sum = 0;
    int count = 0;
    for (const MarkerSegment& segment : segments) {
        if (current.segments.empty()) {
            current.segments.push_back(segment);
            y_sum = segment.center_y;
            count = 1;
            current.center_y = segment.center_y;
            continue;
        }

        if (std::abs(segment.center_y - current.center_y) <= y_tolerance) {
            current.segments.push_back(segment);
            y_sum += segment.center_y;
            ++count;
            current.center_y = static_cast<int>(std::lround(y_sum / static_cast<double>(count)));
            continue;
        }

        if (static_cast<int>(current.segments.size()) >= min_segments) {
            std::sort(current.segments.begin(), current.segments.end(), [](const MarkerSegment& a, const MarkerSegment& b) {
                return a.center_x < b.center_x;
            });
            rows.push_back(current);
        }

        current = MarkerRow{};
        current.segments.push_back(segment);
        y_sum = segment.center_y;
        count = 1;
        current.center_y = segment.center_y;
    }

    if (static_cast<int>(current.segments.size()) >= min_segments) {
        std::sort(current.segments.begin(), current.segments.end(), [](const MarkerSegment& a, const MarkerSegment& b) {
            return a.center_x < b.center_x;
        });
        rows.push_back(current);
    }

    return rows;
}

MarkerColumnSelection chooseBestMarkerColumns(const std::vector<MarkerRow>& rows, int cols)
{
    MarkerColumnSelection best;
    if (cols <= 0) {
        return best;
    }

    for (const MarkerRow& row : rows) {
        if (static_cast<int>(row.segments.size()) < cols) {
            continue;
        }

        for (int start = 0; start + cols <= static_cast<int>(row.segments.size()); ++start) {
            std::vector<MarkerSegment> run(row.segments.begin() + start, row.segments.begin() + start + cols);
            std::vector<double> spacings;
            for (int i = 1; i < static_cast<int>(run.size()); ++i) {
                spacings.push_back(run[i].center_x - run[i - 1].center_x);
            }

            double variance = 0.0;
            if (!spacings.empty()) {
                const double mean = std::accumulate(spacings.begin(), spacings.end(), 0.0) / spacings.size();
                if (mean <= 1.0) {
                    continue;
                }
                for (double spacing : spacings) {
                    const double delta = spacing - mean;
                    variance += delta * delta;
                }
                variance /= spacings.size();
            }

            const int span = run.back().center_x - run.front().center_x;
            if (!best.found || variance < best.spacing_variance ||
                (std::abs(variance - best.spacing_variance) < 1e-6 && span > best.span)) {
                best.found = true;
                best.segments = run;
                best.spacing_variance = variance;
                best.span = span;
            }
        }
    }

    return best;
}

std::vector<int> markerXLines(const std::vector<MarkerSegment>& columns, int cols, int frame_width)
{
    std::vector<int> x_lines;
    if (static_cast<int>(columns.size()) != cols || cols <= 0 || frame_width <= 1) {
        return x_lines;
    }

    if (cols == 1) {
        x_lines.push_back(clampInt(columns.front().left, 0, frame_width - 1));
        x_lines.push_back(clampInt(columns.front().right, 0, frame_width - 1));
        return x_lines;
    }

    std::vector<int> spacings;
    for (int i = 1; i < cols; ++i) {
        spacings.push_back(columns[i].center_x - columns[i - 1].center_x);
    }
    std::sort(spacings.begin(), spacings.end());
    const int spacing = spacings[spacings.size() / 2];
    if (spacing <= 1) {
        return x_lines;
    }

    const int left =
        clampInt(static_cast<int>(std::lround(columns.front().center_x - spacing / 2.0)), 0, frame_width - 2);
    const int right =
        clampInt(static_cast<int>(std::lround(columns.back().center_x + spacing / 2.0)), left + 1, frame_width - 1);

    return fillEvenlySpacedLinesFromBounds(left, right, cols + 1);
}

std::vector<int> markerYLines(const std::vector<MarkerRow>& rows, int top_y, int expected_rows, int frame_height)
{
    std::vector<int> y_lines;
    if (expected_rows <= 0 || static_cast<int>(rows.size()) < expected_rows || frame_height <= 1) {
        return y_lines;
    }

    y_lines.push_back(clampInt(top_y, 0, frame_height - 2));
    for (int i = 0; i < expected_rows; ++i) {
        const int y = clampInt(rows[i].center_y, 0, frame_height - 1);
        if (y <= y_lines.back()) {
            return {};
        }
        y_lines.push_back(y);
    }

    return y_lines;
}

GridDetectionResult detectMarkerBasedWarehouseGrid(const cv::Mat& frame, const GridConfig& config)
{
    if (frame.empty() || config.rows <= 0 || config.cols <= 0) {
        return GridDetectionResult{false, {}};
    }

    const int top_y = detectTopBlackLineY(frame);
    if (top_y < 0) {
        return GridDetectionResult{false, {}};
    }

    std::vector<MarkerSegment> segments = detectWhiteMarkerSegments(frame, top_y, config);
    std::vector<MarkerRow> rows = groupMarkerRows(std::move(segments), config.cols, frame.rows);
    if (static_cast<int>(rows.size()) < config.rows) {
        return GridDetectionResult{false, {}};
    }

    std::sort(rows.begin(), rows.end(), [](const MarkerRow& a, const MarkerRow& b) {
        return a.center_y < b.center_y;
    });
    rows.resize(static_cast<std::size_t>(config.rows));

    const MarkerColumnSelection columns = chooseBestMarkerColumns(rows, config.cols);
    if (!columns.found) {
        return GridDetectionResult{false, {}};
    }

    const std::vector<int> x_lines = markerXLines(columns.segments, config.cols, frame.cols);
    const std::vector<int> y_lines = markerYLines(rows, top_y, config.rows, frame.rows);
    if (static_cast<int>(x_lines.size()) != config.cols + 1 ||
        static_cast<int>(y_lines.size()) != config.rows + 1) {
        return GridDetectionResult{false, {}};
    }

    return buildGridCells(x_lines, y_lines, config);
}

GridDetectionResult smoothGridDetections(
    const GridDetectionResult& previous,
    const GridDetectionResult& current,
    double smoothing_alpha)
{
    if (!previous.found || previous.cells.size() != current.cells.size()) {
        return current;
    }

    const double alpha = std::clamp(smoothing_alpha, 0.0, 1.0);
    GridDetectionResult smoothed = current;

    for (std::size_t cell_index = 0; cell_index < smoothed.cells.size(); ++cell_index) {
        GridCell& smoothed_cell = smoothed.cells[cell_index];
        const GridCell& previous_cell = previous.cells[cell_index];
        const GridCell& current_cell = current.cells[cell_index];

        if (previous_cell.row != current_cell.row ||
            previous_cell.col != current_cell.col ||
            previous_cell.index != current_cell.index ||
            previous_cell.corners.size() != current_cell.corners.size()) {
            return current;
        }

        for (std::size_t corner_index = 0; corner_index < smoothed_cell.corners.size(); ++corner_index) {
            smoothed_cell.corners[corner_index].x = static_cast<float>(
                previous_cell.corners[corner_index].x * (1.0 - alpha) +
                current_cell.corners[corner_index].x * alpha);
            smoothed_cell.corners[corner_index].y = static_cast<float>(
                previous_cell.corners[corner_index].y * (1.0 - alpha) +
                current_cell.corners[corner_index].y * alpha);
        }
    }

    return smoothed;
}

double maxGridCornerDistance(const GridDetectionResult& previous, const GridDetectionResult& current)
{
    if (!previous.found || !current.found || previous.cells.size() != current.cells.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double max_distance = 0.0;
    for (std::size_t cell_index = 0; cell_index < previous.cells.size(); ++cell_index) {
        const GridCell& previous_cell = previous.cells[cell_index];
        const GridCell& current_cell = current.cells[cell_index];

        if (previous_cell.row != current_cell.row ||
            previous_cell.col != current_cell.col ||
            previous_cell.index != current_cell.index ||
            previous_cell.corners.size() != current_cell.corners.size()) {
            return std::numeric_limits<double>::infinity();
        }

        for (std::size_t corner_index = 0; corner_index < previous_cell.corners.size(); ++corner_index) {
            const double dx = static_cast<double>(
                current_cell.corners[corner_index].x - previous_cell.corners[corner_index].x);
            const double dy = static_cast<double>(
                current_cell.corners[corner_index].y - previous_cell.corners[corner_index].y);
            max_distance = std::max(max_distance, std::sqrt(dx * dx + dy * dy));
        }
    }

    return max_distance;
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

GridDetectionResult completeGridFromBounds(
    const std::vector<int>& x_lines,
    const std::vector<int>& y_lines,
    const GridConfig& config)
{
    if (!config.enabled || config.rows <= 0 || config.cols <= 0) {
        return GridDetectionResult{false, {}};
    }

    const GridBoundsCandidate candidate = chooseBestGridBoundsCandidate(x_lines, y_lines, config);
    if (!candidate.found) {
        return GridDetectionResult{false, {}};
    }

    const std::vector<int> completed_x =
        fillEvenlySpacedLinesFromBounds(candidate.left, candidate.right, config.cols + 1);
    const std::vector<int> completed_y =
        fillEvenlySpacedLinesFromBounds(candidate.top, candidate.bottom, config.rows + 1);
    return buildGridCells(completed_x, completed_y, config);
}

GridDetectionResult detectWarehouseGrid(const cv::Mat& frame, const GridConfig& config)
{
    if (!config.enabled || config.rows <= 0 || config.cols <= 0 || frame.empty()) {
        return GridDetectionResult{false, {}};
    }

    const GridDetectionResult marker_grid = detectMarkerBasedWarehouseGrid(frame, config);
    if (marker_grid.found) {
        return marker_grid;
    }

    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);
    cv::Canny(blurred, edges, 50.0, 150.0, 3);

    std::vector<cv::Vec4i> lines;
    const int min_dimension = std::min(frame.cols, frame.rows);
    const int min_line_length = std::max(20, min_dimension / 5);
    cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0, 40, min_line_length, 12.0);

    std::vector<int> x_positions;
    std::vector<int> y_positions;
    x_positions.reserve(lines.size());
    y_positions.reserve(lines.size());

    for (const cv::Vec4i& line : lines) {
        const int x1 = line[0];
        const int y1 = line[1];
        const int x2 = line[2];
        const int y2 = line[3];
        const int dx = std::abs(x2 - x1);
        const int dy = std::abs(y2 - y1);

        if (dy >= min_line_length && dx <= std::max(3, dy / 5)) {
            x_positions.push_back((x1 + x2) / 2);
        } else if (dx >= min_line_length && dy <= std::max(3, dx / 5)) {
            y_positions.push_back((y1 + y2) / 2);
        }
    }

    const int x_tolerance = std::max(6, frame.cols / std::max(24, config.cols * 8));
    const int y_tolerance = std::max(6, frame.rows / std::max(24, config.rows * 8));
    const std::vector<int> clustered_x = clusterLinePositions(x_positions, x_tolerance);
    const std::vector<int> clustered_y = clusterLinePositions(y_positions, y_tolerance);
    const std::vector<int> x_lines = chooseGridLinePositions(clustered_x, config.cols + 1);
    const std::vector<int> y_lines = chooseGridLinePositions(clustered_y, config.rows + 1);

    GridDetectionResult grid = buildGridCells(x_lines, y_lines, config);
    if (grid.found && gridAspectMatches(x_lines, y_lines, config)) {
        return grid;
    }

    return completeGridFromBounds(clustered_x, clustered_y, config);
}

GridDetectionResult updateGridTracker(
    const GridDetectionResult& raw_grid,
    GridTrackerState& state,
    const GridTrackerConfig& config)
{
    if (raw_grid.found) {
        if (state.last_good.found &&
            maxGridCornerDistance(state.last_good, raw_grid) <= config.jitter_hold_threshold) {
            state.missed_frames = 0;
            return state.last_good;
        }

        state.last_good = smoothGridDetections(state.last_good, raw_grid, config.smoothing_alpha);
        state.missed_frames = 0;
        return state.last_good;
    }

    ++state.missed_frames;
    if (state.last_good.found && state.missed_frames <= config.cache_frames) {
        return state.last_good;
    }

    return GridDetectionResult{false, {}};
}

const GridCell* findContainingCell(const std::vector<GridCell>& cells, cv::Point2f point)
{
    for (const GridCell& cell : cells) {
        if (cell.corners.size() < 3) {
            continue;
        }

        if (cv::pointPolygonTest(cell.corners, point, false) >= 0.0) {
            return &cell;
        }
    }

    return nullptr;
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
    drawOverlay(frame, metrics, detections, GridDetectionResult{false, {}});
}

void drawOverlay(
    cv::Mat& frame,
    const FrameMetrics& metrics,
    const std::vector<DetectionResult>& detections,
    const GridDetectionResult& grid)
{
    if (frame.empty()) {
        return;
    }

    const cv::Point center(metrics.center_x, metrics.center_y);
    const cv::Scalar center_color(0, 255, 0);
    const cv::Scalar ball_color(0, 0, 255);
    const cv::Scalar other_ball_color(255, 255, 0);
    const cv::Scalar line_color(0, 255, 255);
    const cv::Scalar grid_color(0, 255, 255);

    if (grid.found) {
        std::vector<int> drawn_cell_indices;
        for (const DetectionResult& detection : detections) {
            if (!detection.found) {
                continue;
            }

            const GridCell* cell = findContainingCell(
                grid.cells,
                cv::Point2f(static_cast<float>(detection.x), static_cast<float>(detection.y)));
            if (cell == nullptr) {
                continue;
            }

            if (std::find(drawn_cell_indices.begin(), drawn_cell_indices.end(), cell->index) !=
                drawn_cell_indices.end()) {
                continue;
            }
            drawn_cell_indices.push_back(cell->index);

            std::vector<cv::Point> corners;
            corners.reserve(cell->corners.size());
            for (const cv::Point2f& corner : cell->corners) {
                corners.emplace_back(
                    static_cast<int>(std::lround(corner.x)),
                    static_cast<int>(std::lround(corner.y)));
            }

            if (corners.size() >= 3) {
                cv::polylines(frame, corners, true, grid_color, 2, cv::LINE_8);
                cv::putText(
                    frame,
                    std::to_string(cell->index),
                    corners.front() + cv::Point(6, 18),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    grid_color,
                    2,
                    cv::LINE_8);
            }
        }
    }

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

    for (int index : buildCameraScanOrder(scan_max)) {
        cv::VideoCapture capture;
        if (!capture.open(index)) {
            continue;
        }

        if (readValidFrame(capture)) {
            candidates.push_back(CameraCandidate{index, true, isLikelyExternalCameraIndex(index)});
            capture.release();
            if (candidates.back().likely_external) {
                break;
            }
        } else {
            capture.release();
        }
    }

    return candidates;
}

std::vector<int> buildCameraScanOrder(int scan_max)
{
    std::vector<int> order;
    if (scan_max < 0) {
        return order;
    }

#ifdef _WIN32
    if (scan_max >= 1) {
        order.push_back(1);
    }
    order.push_back(0);
    for (int index = 2; index <= scan_max; ++index) {
        order.push_back(index);
    }
#else
    for (int index = 0; index <= scan_max; ++index) {
        order.push_back(index);
    }
#endif

    return order;
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
    GridTrackerState grid_tracker_state;
    const GridTrackerConfig grid_tracker_config{options.grid_cache_frames, 0.35};
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
            const GridDetectionResult raw_grid = options.grid.enabled
                ? detectWarehouseGrid(frame, options.grid)
                : GridDetectionResult{false, {}};
            const GridDetectionResult grid = options.grid.enabled
                ? updateGridTracker(raw_grid, grid_tracker_state, grid_tracker_config)
                : raw_grid;
            drawOverlay(frame, metrics, detections, grid);
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
