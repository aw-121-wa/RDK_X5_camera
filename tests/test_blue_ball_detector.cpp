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

void test_selects_largest_blue_region()
{
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(frame, cv::Point(100, 100), 15, cv::Scalar(255, 0, 0), -1);
    cv::circle(frame, cv::Point(420, 330), 45, cv::Scalar(255, 0, 0), -1);

    const DetectionResult result = detectBlueBall(frame, defaultBlueRange(), 300.0);

    assert(result.found);
    assert(std::abs(result.x - 420) <= 1);
    assert(std::abs(result.y - 330) <= 1);
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

} // namespace

int main()
{
    test_detects_blue_circle_center();
    test_missing_blue_circle_returns_not_found();
    test_selects_largest_blue_region();
    test_computes_frame_metrics_for_found_ball();
    test_computes_frame_metrics_for_missing_ball();
    test_formats_csv_output_for_lower_controller();
    test_formats_missing_csv_output();
    return 0;
}
