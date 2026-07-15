#ifndef BLUE_BALL_DETECTOR_HPP
#define BLUE_BALL_DETECTOR_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct HSVRange {
    int h_min;
    int h_max;
    int s_min;
    int s_max;
    int v_min;
    int v_max;
};

struct DetectionResult {
    bool found;
    int x;
    int y;
    double area;
};

struct FrameMetrics {
    bool found;
    int ball_x;
    int ball_y;
    int center_x;
    int center_y;
    int dx;
    int dy;
    double distance;
};

struct CameraCandidate {
    int index;
    bool readable;
    bool likely_external;
};

struct GridConfig {
    bool enabled;
    int rows;
    int cols;
};

struct GridCell {
    int row;
    int col;
    int index;
    std::vector<cv::Point2f> corners;
};

struct GridDetectionResult {
    bool found;
    std::vector<GridCell> cells;
};

struct GridTrackerConfig {
    int cache_frames;
    double smoothing_alpha;
};

struct GridTrackerState {
    GridDetectionResult last_good{false, {}};
    int missed_frames = 0;
};

DetectionResult detectBlueBall(const cv::Mat& frame, const HSVRange& hsv, double min_area);

std::vector<DetectionResult> detectBlueBalls(const cv::Mat& frame, const HSVRange& hsv, double min_area);

GridDetectionResult detectWarehouseGrid(const cv::Mat& frame, const GridConfig& config);

GridDetectionResult completeGridFromBounds(
    const std::vector<int>& x_lines,
    const std::vector<int>& y_lines,
    const GridConfig& config);

GridDetectionResult updateGridTracker(
    const GridDetectionResult& raw_grid,
    GridTrackerState& state,
    const GridTrackerConfig& config);

const GridCell* findContainingCell(const std::vector<GridCell>& cells, cv::Point2f point);

FrameMetrics computeFrameMetrics(const cv::Size& frame_size, const DetectionResult& detection);

std::string formatMetricsCsv(const FrameMetrics& metrics);

void drawOverlay(cv::Mat& frame, const FrameMetrics& metrics);

void drawOverlay(cv::Mat& frame, const FrameMetrics& metrics, const std::vector<DetectionResult>& detections);

void drawOverlay(
    cv::Mat& frame,
    const FrameMetrics& metrics,
    const std::vector<DetectionResult>& detections,
    const GridDetectionResult& grid);

std::vector<CameraCandidate> scanCameraCandidates(int scan_max);

std::vector<int> buildCameraScanOrder(int scan_max);

int choosePreferredCameraIndex(const std::vector<CameraCandidate>& candidates);

int findCameraIndex(int scan_max);

int runBlueBallDetector(int argc, char** argv);

#endif
