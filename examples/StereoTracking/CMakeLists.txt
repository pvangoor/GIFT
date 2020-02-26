# find_package(GIFT)
find_package(OpenCV 3 REQUIRED)

add_executable(StereoTracking main.cpp)

target_include_directories(StereoTracking PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(StereoTracking GIFT ${OpenCV_LIBS})