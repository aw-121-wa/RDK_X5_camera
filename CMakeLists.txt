cmake_minimum_required(VERSION 3.10)

project(blue_ball_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_library(blue_ball_core
    src/blue_ball_detector.cpp
)

target_include_directories(blue_ball_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(blue_ball_core
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(blue_ball_detector
    src/main.cpp
)

target_link_libraries(blue_ball_detector
    PRIVATE
        blue_ball_core
)

enable_testing()

add_executable(blue_ball_detector_tests
    tests/test_blue_ball_detector.cpp
)

target_link_libraries(blue_ball_detector_tests
    PRIVATE
        blue_ball_core
)

add_test(NAME blue_ball_detector_tests COMMAND blue_ball_detector_tests)
