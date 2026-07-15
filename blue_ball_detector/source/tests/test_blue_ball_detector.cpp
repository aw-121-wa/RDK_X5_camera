#include "blue_ball_detector.hpp"

#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

HSVRange defaultBlueRange()
{
    return HSVRange{90, 130, 60, 255, 40, 255};
}

void test_detects_blue_circle_center()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(300, 200), 35, cv::Scalar(255, 0, 0), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(result.found);
    assert(std::abs(result.x - 300) <= 1);
    assert(std::abs(result.y - 200) <= 1);
    assert(result.area > 300.0);
}

void test_missing_blue_circle_returns_not_found()
{
    cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(!result.found);
    assert(result.x == -1);
    assert(result.y == -1);
    assert(result.area == 0.0);
}

void test_detects_blue_region_when_largest_is_rightmost()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(100, 100), 15, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(420, 330), 45, cv::Scalar(255, 0, 0), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(result.found);
    assert(std::abs(result.x - 420) <= 1);
    assert(std::abs(result.y - 330) <= 1);
}

void test_detects_all_blue_regions()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(110, 120), 24, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(310, 220), 30, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(510, 320), 36, cv::Scalar(255, 0, 0), -1);

    const std::vector<DetectionResult> results = detectBlueBalls(frame, defaultBlueRange(), 300.0);

    assert(results.size() == 3);
    assert(std::abs(results[0].x - 110) <= 1);
    assert(std::abs(results[1].x - 310) <= 1);
    assert(std::abs(results[2].x - 510) <= 1);
}

void test_detects_less_saturated_small_blue_ball()
{
    cv::Mat frame(160, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(90, 70), 8, cv::Scalar(150, 110, 110), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 100.0);

    assert(result.found);
    assert(std::abs(result.x - 90) <= 1);
    assert(std::abs(result.y - 70) <= 1);
}

void test_selects_rightmost_blue_region_not_largest()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(160, 260), 55, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(500, 180), 24, cv::Scalar(255, 0, 0), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(result.found);
    assert(std::abs(result.x - 500) <= 1);
    assert(std::abs(result.y - 180) <= 1);
}

void test_selects_larger_region_when_rightmost_x_ties()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(420, 130), 18, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(420, 340), 34, cv::Scalar(255, 0, 0), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(result.found);
    assert(std::abs(result.x - 420) <= 1);
    assert(std::abs(result.y - 340) <= 1);
}

void test_computes_frame_metrics_for_found_ball()
{
    const DetectionResult detection{true, 320, 210, 1200.0};

    const FrameMetrics metrics = computeFrameMetrics(cv::Size(640, 480), detection);

    assert(metrics.found);
    assert(metrics.ball_x == 320);
    assert(metrics.ball_y == 210);
    assert(metrics.center_x == 320);
    assert(metrics.center_y == 240);
    assert(metrics.dx == 0);
    assert(metrics.dy == -30);
    assert(std::abs(metrics.distance - 30.0) < 0.001);
}

void test_computes_frame_metrics_for_missing_ball()
{
    const DetectionResult detection{false, -1, -1, 0.0};

    const FrameMetrics metrics = computeFrameMetrics(cv::Size(320, 240), detection);

    assert(!metrics.found);
    assert(metrics.ball_x == -1);
    assert(metrics.ball_y == -1);
    assert(metrics.center_x == 160);
    assert(metrics.center_y == 120);
    assert(metrics.dx == 0);
    assert(metrics.dy == 0);
    assert(metrics.distance == -1.0);
}

std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;

    while (std::getline(stream, part, ',')) {
        parts.push_back(part);
    }

    return parts;
}

void test_formats_csv_output_for_lower_controller()
{
    const FrameMetrics metrics{true, 320, 210, 320, 240, 0, -30, 30.0};

    const std::string line = formatMetricsCsv(metrics);
    const std::vector<std::string> parts = splitCsv(line);

    assert(line == "B,1,320,210,320,240,0,-30,30.00");
    assert(parts.size() == 9);
}

void test_formats_missing_csv_output()
{
    const FrameMetrics metrics{false, -1, -1, 320, 240, 0, 0, -1.0};

    const std::string line = formatMetricsCsv(metrics);
    const std::vector<std::string> parts = splitCsv(line);

    assert(line == "B,0,-1,-1,320,240,0,0,-1.00");
    assert(parts.size() == 9);
}

void test_draw_overlay_marks_center_and_ball()
{
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const FrameMetrics metrics{true, 70, 20, 50, 50, 20, -30, std::sqrt(1300.0)};

    drawOverlay(frame, metrics);

    const cv::Vec3b center_pixel = frame.at<cv::Vec3b>(50, 50);
    const cv::Vec3b ball_pixel = frame.at<cv::Vec3b>(20, 70);

    assert(center_pixel[1] > 200);
    assert(center_pixel[0] < 50);
    assert(center_pixel[2] < 50);
    assert(ball_pixel[2] > 200);
    assert(ball_pixel[0] < 50);
    assert(ball_pixel[1] < 50);
}

void test_draw_overlay_marks_center_when_ball_missing()
{
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const FrameMetrics metrics{false, -1, -1, 50, 50, 0, 0, -1.0};

    drawOverlay(frame, metrics);

    const cv::Vec3b center_pixel = frame.at<cv::Vec3b>(50, 50);
    const cv::Vec3b background_pixel = frame.at<cv::Vec3b>(20, 70);

    assert(center_pixel[1] > 200);
    assert(center_pixel[0] < 50);
    assert(center_pixel[2] < 50);
    assert(background_pixel == cv::Vec3b(0, 0, 0));
}

void test_draw_overlay_marks_all_detected_balls()
{
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const FrameMetrics metrics{true, 80, 70, 50, 50, 30, 20, std::sqrt(1300.0)};
    const std::vector<DetectionResult> detections{
        DetectionResult{true, 20, 30, 900.0},
        DetectionResult{true, 80, 70, 900.0},
    };

    drawOverlay(frame, metrics, detections);

    const cv::Vec3b other_ball_pixel = frame.at<cv::Vec3b>(30, 20);
    const cv::Vec3b selected_ball_pixel = frame.at<cv::Vec3b>(70, 80);

    assert(other_ball_pixel[0] > 200);
    assert(other_ball_pixel[1] > 200);
    assert(other_ball_pixel[2] < 50);
    assert(selected_ball_pixel[2] > 200);
    assert(selected_ball_pixel[0] < 50);
    assert(selected_ball_pixel[1] < 50);
}

void drawSyntheticGrid(cv::Mat& frame, int rows, int cols)
{
    const int width = frame.cols;
    const int height = frame.rows;

    for (int col = 0; col <= cols; ++col) {
        const int x = (width - 1) * col / cols;
        cv::line(frame, cv::Point(x, 0), cv::Point(x, height - 1), cv::Scalar(255, 255, 255), 3);
    }

    for (int row = 0; row <= rows; ++row) {
        const int y = (height - 1) * row / rows;
        cv::line(frame, cv::Point(0, y), cv::Point(width - 1, y), cv::Scalar(255, 255, 255), 3);
    }
}

void drawSyntheticMarkerGrid(cv::Mat& frame, int rows, int cols)
{
    frame.setTo(cv::Scalar(40, 40, 40));

    const int top = 20;
    const int bottom = frame.rows - 20;
    cv::line(frame, cv::Point(10, top), cv::Point(frame.cols - 11, top), cv::Scalar(0, 0, 0), 5);

    const int cell_width = frame.cols / cols;
    const int marker_width = std::max(16, cell_width - 18);
    for (int row = 0; row < rows; ++row) {
        const int y = top + (bottom - top) * (row + 1) / rows;
        for (int col = 0; col < cols; ++col) {
            const int center_x = cell_width * col + cell_width / 2;
            cv::line(
                frame,
                cv::Point(center_x - marker_width / 2, y),
                cv::Point(center_x + marker_width / 2, y),
                cv::Scalar(255, 255, 255),
                4);
        }
    }
}

void test_detects_marker_based_grid_from_black_top_line_and_white_bottom_segments()
{
    cv::Mat frame(140, 240, CV_8UC3);
    drawSyntheticMarkerGrid(frame, 2, 3);

    const GridDetectionResult grid = detectWarehouseGrid(frame, GridConfig{true, 2, 3});
    const GridCell* cell = findContainingCell(grid.cells, cv::Point2f(120.0F, 45.0F));

    assert(grid.found);
    assert(grid.cells.size() == 6);
    assert(cell != nullptr);
    assert(cell->row == 0);
    assert(cell->col == 1);
    assert(cell->index == 2);
}

void test_marker_grid_rejects_mismatched_white_segment_count()
{
    cv::Mat frame(140, 240, CV_8UC3);
    drawSyntheticMarkerGrid(frame, 2, 3);
    cv::rectangle(frame, cv::Rect(165, 62, 70, 16), cv::Scalar(40, 40, 40), -1);

    const GridDetectionResult grid = detectWarehouseGrid(frame, GridConfig{true, 2, 3});

    assert(!grid.found);
}

void test_marker_grid_takes_priority_over_misleading_full_grid_lines()
{
    cv::Mat frame(170, 300, CV_8UC3);
    drawSyntheticMarkerGrid(frame, 2, 3);

    for (int x : {25, 125, 225, 285}) {
        cv::line(frame, cv::Point(x, 40), cv::Point(x, 160), cv::Scalar(255, 255, 255), 2);
    }
    for (int y : {40, 100, 160}) {
        cv::line(frame, cv::Point(25, y), cv::Point(285, y), cv::Scalar(255, 255, 255), 2);
    }

    const GridDetectionResult grid = detectWarehouseGrid(frame, GridConfig{true, 2, 3});
    const GridCell* marker_cell = findContainingCell(grid.cells, cv::Point2f(150.0F, 30.0F));

    assert(grid.found);
    assert(marker_cell != nullptr);
    assert(marker_cell->row == 0);
    assert(marker_cell->col == 1);
}

void test_detects_grid_cells_in_reading_order()
{
    cv::Mat frame(120, 240, CV_8UC3, cv::Scalar(0, 0, 0));
    drawSyntheticGrid(frame, 2, 3);

    const GridDetectionResult grid = detectWarehouseGrid(frame, GridConfig{true, 2, 3});

    assert(grid.found);
    assert(grid.cells.size() == 6);
    assert(grid.cells[0].row == 0);
    assert(grid.cells[0].col == 0);
    assert(grid.cells[0].index == 1);
    assert(grid.cells[2].row == 0);
    assert(grid.cells[2].col == 2);
    assert(grid.cells[2].index == 3);
    assert(grid.cells[3].row == 1);
    assert(grid.cells[3].col == 0);
    assert(grid.cells[3].index == 4);
    assert(grid.cells[5].row == 1);
    assert(grid.cells[5].col == 2);
    assert(grid.cells[5].index == 6);
}

void test_finds_cell_containing_ball_center()
{
    cv::Mat frame(120, 240, CV_8UC3, cv::Scalar(0, 0, 0));
    drawSyntheticGrid(frame, 2, 3);

    const GridDetectionResult grid = detectWarehouseGrid(frame, GridConfig{true, 2, 3});
    const GridCell* cell = findContainingCell(grid.cells, cv::Point2f(120.0F, 30.0F));

    assert(cell != nullptr);
    assert(cell->row == 0);
    assert(cell->col == 1);
    assert(cell->index == 2);
}

void test_draw_overlay_marks_blue_ball_cell_with_yellow_box()
{
    cv::Mat frame(100, 120, CV_8UC3, cv::Scalar(0, 0, 0));
    const GridDetectionResult grid{
        true,
        std::vector<GridCell>{
            GridCell{0, 0, 1, std::vector<cv::Point2f>{
                cv::Point2f(0.0F, 0.0F),
                cv::Point2f(59.0F, 0.0F),
                cv::Point2f(59.0F, 49.0F),
                cv::Point2f(0.0F, 49.0F)}},
            GridCell{0, 1, 2, std::vector<cv::Point2f>{
                cv::Point2f(60.0F, 0.0F),
                cv::Point2f(119.0F, 0.0F),
                cv::Point2f(119.0F, 49.0F),
                cv::Point2f(60.0F, 49.0F)}}}};
    const FrameMetrics metrics{true, 80, 20, 60, 50, 20, -30, std::sqrt(1300.0)};
    const std::vector<DetectionResult> detections{DetectionResult{true, 80, 20, 900.0}};

    drawOverlay(frame, metrics, detections, grid);

    const cv::Vec3b yellow_border_pixel = frame.at<cv::Vec3b>(0, 60);
    assert(yellow_border_pixel[0] < 50);
    assert(yellow_border_pixel[1] > 200);
    assert(yellow_border_pixel[2] > 200);
}

void test_grid_detection_fails_for_empty_frame()
{
    const GridDetectionResult grid = detectWarehouseGrid(cv::Mat(), GridConfig{true, 2, 3});

    assert(!grid.found);
    assert(grid.cells.empty());
}

GridDetectionResult makeTwoByTwoGrid(float offset_x)
{
    return GridDetectionResult{
        true,
        std::vector<GridCell>{
            GridCell{0, 0, 1, std::vector<cv::Point2f>{
                cv::Point2f(offset_x + 0.0F, 0.0F),
                cv::Point2f(offset_x + 50.0F, 0.0F),
                cv::Point2f(offset_x + 50.0F, 50.0F),
                cv::Point2f(offset_x + 0.0F, 50.0F)}},
            GridCell{0, 1, 2, std::vector<cv::Point2f>{
                cv::Point2f(offset_x + 50.0F, 0.0F),
                cv::Point2f(offset_x + 100.0F, 0.0F),
                cv::Point2f(offset_x + 100.0F, 50.0F),
                cv::Point2f(offset_x + 50.0F, 50.0F)}},
            GridCell{1, 0, 3, std::vector<cv::Point2f>{
                cv::Point2f(offset_x + 0.0F, 50.0F),
                cv::Point2f(offset_x + 50.0F, 50.0F),
                cv::Point2f(offset_x + 50.0F, 100.0F),
                cv::Point2f(offset_x + 0.0F, 100.0F)}},
            GridCell{1, 1, 4, std::vector<cv::Point2f>{
                cv::Point2f(offset_x + 50.0F, 50.0F),
                cv::Point2f(offset_x + 100.0F, 50.0F),
                cv::Point2f(offset_x + 100.0F, 100.0F),
                cv::Point2f(offset_x + 50.0F, 100.0F)}}}};
}

void test_grid_tracker_updates_cache_when_grid_is_found()
{
    GridTrackerState state;
    const GridTrackerConfig config{15, 0.5};
    const GridDetectionResult raw = makeTwoByTwoGrid(0.0F);

    const GridDetectionResult tracked = updateGridTracker(raw, state, config);

    assert(tracked.found);
    assert(state.last_good.found);
    assert(state.missed_frames == 0);
    assert(tracked.cells.size() == 4);
}

void test_grid_tracker_reuses_recent_grid_when_frame_detection_fails()
{
    GridTrackerState state;
    const GridTrackerConfig config{2, 0.5};
    updateGridTracker(makeTwoByTwoGrid(0.0F), state, config);

    const GridDetectionResult tracked = updateGridTracker(GridDetectionResult{false, {}}, state, config);

    assert(tracked.found);
    assert(tracked.cells.size() == 4);
    assert(state.missed_frames == 1);
}

void test_grid_tracker_expires_cached_grid_after_limit()
{
    GridTrackerState state;
    const GridTrackerConfig config{2, 0.5};
    updateGridTracker(makeTwoByTwoGrid(0.0F), state, config);

    updateGridTracker(GridDetectionResult{false, {}}, state, config);
    updateGridTracker(GridDetectionResult{false, {}}, state, config);
    const GridDetectionResult expired = updateGridTracker(GridDetectionResult{false, {}}, state, config);

    assert(!expired.found);
    assert(expired.cells.empty());
    assert(state.missed_frames == 3);
}

void test_grid_tracker_smooths_successive_grid_positions()
{
    GridTrackerState state;
    const GridTrackerConfig config{15, 0.5};
    updateGridTracker(makeTwoByTwoGrid(0.0F), state, config);

    const GridDetectionResult tracked = updateGridTracker(makeTwoByTwoGrid(10.0F), state, config);

    assert(tracked.found);
    assert(std::abs(tracked.cells[0].corners[0].x - 5.0F) < 0.001F);
}

void test_completes_grid_from_outer_bounds_when_inner_lines_are_missing()
{
    const std::vector<int> x_lines{0, 133};
    const std::vector<int> y_lines{0, 100};

    const GridDetectionResult grid = completeGridFromBounds(x_lines, y_lines, GridConfig{true, 2, 2});

    assert(grid.found);
    assert(grid.cells.size() == 4);
    assert(std::abs(grid.cells[1].corners[0].x - 67.0F) < 0.001F);
    assert(grid.cells[3].index == 4);
}

void test_grid_config_defaults_to_four_to_three_cell_aspect()
{
    const GridConfig config{true, 2, 3};

    assert(std::abs(config.cell_aspect - (4.0 / 3.0)) < 0.001);
    assert(std::abs(config.aspect_tolerance - 0.35) < 0.001);
}

void test_grid_tracker_keeps_previous_grid_for_small_jitter()
{
    GridTrackerState state;
    const GridTrackerConfig config{15, 0.5, 5.0};
    updateGridTracker(makeTwoByTwoGrid(0.0F), state, config);

    const GridDetectionResult tracked = updateGridTracker(makeTwoByTwoGrid(2.0F), state, config);

    assert(tracked.found);
    assert(std::abs(tracked.cells[0].corners[0].x - 0.0F) < 0.001F);
    assert(state.missed_frames == 0);
}

void test_completes_grid_from_bounds_when_aspect_matches_cell_ratio()
{
    const GridConfig config{true, 2, 3, 2.0, 0.10};
    const std::vector<int> x_lines{0, 300};
    const std::vector<int> y_lines{0, 100};

    const GridDetectionResult grid = completeGridFromBounds(x_lines, y_lines, config);

    assert(grid.found);
    assert(grid.cells.size() == 6);
    assert(std::abs(grid.cells[1].corners[0].x - 100.0F) < 0.001F);
    assert(std::abs(grid.cells[5].corners[2].x - 300.0F) < 0.001F);
    assert(std::abs(grid.cells[5].corners[2].y - 100.0F) < 0.001F);
}

void test_rejects_grid_bounds_when_aspect_is_too_far_from_cell_ratio()
{
    const GridConfig config{true, 2, 3, 2.0, 0.10};
    const std::vector<int> x_lines{0, 100};
    const std::vector<int> y_lines{0, 100};

    const GridDetectionResult grid = completeGridFromBounds(x_lines, y_lines, config);

    assert(!grid.found);
    assert(grid.cells.empty());
}

void test_chooses_grid_bounds_with_best_matching_aspect()
{
    const GridConfig config{true, 2, 3, 2.0, 0.20};
    const std::vector<int> x_lines{0, 300, 320};
    const std::vector<int> y_lines{0, 100};

    const GridDetectionResult grid = completeGridFromBounds(x_lines, y_lines, config);

    assert(grid.found);
    assert(std::abs(grid.cells[5].corners[2].x - 300.0F) < 0.001F);
}

void test_run_rejects_invalid_cell_aspect_argument()
{
    const char* args[] = {"blue_ball_detector", "--cell-aspect", "0"};
    char* argv[] = {
        const_cast<char*>(args[0]),
        const_cast<char*>(args[1]),
        const_cast<char*>(args[2])};

    assert(runBlueBallDetector(3, argv) == 2);
}

void test_run_rejects_invalid_grid_aspect_tolerance_argument()
{
    const char* args[] = {"blue_ball_detector", "--grid-aspect-tolerance", "0"};
    char* argv[] = {
        const_cast<char*>(args[0]),
        const_cast<char*>(args[1]),
        const_cast<char*>(args[2])};

    assert(runBlueBallDetector(3, argv) == 2);
}

void test_camera_scan_order_prioritizes_common_external_camera_on_windows()
{
    const std::vector<int> order = buildCameraScanOrder(3);

#ifdef _WIN32
    assert(order == std::vector<int>({1, 0, 2, 3}));
#else
    assert(order == std::vector<int>({0, 1, 2, 3}));
#endif
}

void test_camera_selection_uses_only_available_camera()
{
    const std::vector<CameraCandidate> candidates{
        CameraCandidate{2, true, false},
    };

    assert(choosePreferredCameraIndex(candidates) == 2);
}

void test_camera_selection_prefers_external_over_index_zero()
{
    const std::vector<CameraCandidate> candidates{
        CameraCandidate{0, true, false},
        CameraCandidate{1, true, true},
    };

    assert(choosePreferredCameraIndex(candidates) == 1);
}

void test_camera_selection_uses_lowest_external_index()
{
    const std::vector<CameraCandidate> candidates{
        CameraCandidate{0, true, false},
        CameraCandidate{3, true, true},
        CameraCandidate{1, true, true},
    };

    assert(choosePreferredCameraIndex(candidates) == 1);
}

void test_camera_selection_falls_back_to_lowest_readable_index()
{
    const std::vector<CameraCandidate> candidates{
        CameraCandidate{4, true, false},
        CameraCandidate{2, true, false},
    };

    assert(choosePreferredCameraIndex(candidates) == 2);
}

void test_camera_selection_returns_negative_when_empty()
{
    const std::vector<CameraCandidate> candidates;

    assert(choosePreferredCameraIndex(candidates) == -1);
}

} // namespace

int main()
{
    test_detects_blue_circle_center();
    test_missing_blue_circle_returns_not_found();
    test_detects_blue_region_when_largest_is_rightmost();
    test_detects_all_blue_regions();
    test_detects_less_saturated_small_blue_ball();
    test_selects_rightmost_blue_region_not_largest();
    test_selects_larger_region_when_rightmost_x_ties();
    test_computes_frame_metrics_for_found_ball();
    test_computes_frame_metrics_for_missing_ball();
    test_formats_csv_output_for_lower_controller();
    test_formats_missing_csv_output();
    test_draw_overlay_marks_center_and_ball();
    test_draw_overlay_marks_center_when_ball_missing();
    test_draw_overlay_marks_all_detected_balls();
    test_detects_marker_based_grid_from_black_top_line_and_white_bottom_segments();
    test_marker_grid_rejects_mismatched_white_segment_count();
    test_marker_grid_takes_priority_over_misleading_full_grid_lines();
    test_detects_grid_cells_in_reading_order();
    test_finds_cell_containing_ball_center();
    test_draw_overlay_marks_blue_ball_cell_with_yellow_box();
    test_grid_detection_fails_for_empty_frame();
    test_grid_tracker_updates_cache_when_grid_is_found();
    test_grid_tracker_reuses_recent_grid_when_frame_detection_fails();
    test_grid_tracker_expires_cached_grid_after_limit();
    test_grid_tracker_smooths_successive_grid_positions();
    test_completes_grid_from_outer_bounds_when_inner_lines_are_missing();
    test_grid_config_defaults_to_four_to_three_cell_aspect();
    test_grid_tracker_keeps_previous_grid_for_small_jitter();
    test_completes_grid_from_bounds_when_aspect_matches_cell_ratio();
    test_rejects_grid_bounds_when_aspect_is_too_far_from_cell_ratio();
    test_chooses_grid_bounds_with_best_matching_aspect();
    test_run_rejects_invalid_cell_aspect_argument();
    test_run_rejects_invalid_grid_aspect_tolerance_argument();
    test_camera_scan_order_prioritizes_common_external_camera_on_windows();
    test_camera_selection_uses_only_available_camera();
    test_camera_selection_prefers_external_over_index_zero();
    test_camera_selection_uses_lowest_external_index();
    test_camera_selection_falls_back_to_lowest_readable_index();
    test_camera_selection_returns_negative_when_empty();
    return 0;
}
