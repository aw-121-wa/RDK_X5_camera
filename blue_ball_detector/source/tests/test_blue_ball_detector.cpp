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
    return HSVRange{90, 130, 80, 255, 50, 255};
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

} // namespace

int main()
{
    test_detects_blue_circle_center();
    test_missing_blue_circle_returns_not_found();
    test_detects_blue_region_when_largest_is_rightmost();
    test_detects_all_blue_regions();
    test_selects_rightmost_blue_region_not_largest();
    test_selects_larger_region_when_rightmost_x_ties();
    test_computes_frame_metrics_for_found_ball();
    test_computes_frame_metrics_for_missing_ball();
    test_formats_csv_output_for_lower_controller();
    test_formats_missing_csv_output();
    test_draw_overlay_marks_center_and_ball();
    test_draw_overlay_marks_center_when_ball_missing();
    test_draw_overlay_marks_all_detected_balls();
    return 0;
}
