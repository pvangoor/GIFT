enable_testing()
find_package(GTest REQUIRED)

add_executable(test_EgoMotion test_EgoMotion.cpp)
add_executable(test_ImagePyramid test_ImagePyramid.cpp)
add_executable(test_OptimiseParameters test_OptimiseParameters.cpp)

target_include_directories(test_EgoMotion PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_EgoMotion
GTest::GTest
GTest::Main
GIFT
)

target_include_directories(test_ImagePyramid PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_ImagePyramid
GTest::GTest
GTest::Main
GIFT
)

target_include_directories(test_OptimiseParameters PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_OptimiseParameters
GTest::GTest
GTest::Main
GIFT
)

add_test(test_EgoMotion test_EgoMotion)
add_test(test_ImagePyramid test_ImagePyramid)
add_test(test_OptimiseParameters test_OptimiseParameters)