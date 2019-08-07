enable_testing()
find_package(GTest REQUIRED)

add_executable(test_EgoMotion test_EgoMotion.cpp)

target_include_directories(test_EgoMotion PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_EgoMotion
GTest::GTest
GTest::Main
GIFT
)

add_test(test_EgoMotion test_EgoMotion)