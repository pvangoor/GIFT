find_package(GTest REQUIRED)

# Link the tests with the test data
add_compile_definitions(TEST_DATA_DIR="${TEST_DATA_DIR}/")

set(TESTS_LIST
    test_EgoMotion
    test_ImagePyramid
    test_OptimiseParameters
    test_PatchFeatureTracker
    test_KeyPointFeatureTracker
)

foreach(TEST_NAME ${TESTS_LIST})
    add_executable(${TEST_NAME} "${TEST_NAME}.cpp")
    target_include_directories(${TEST_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME}
        GTest::GTest
        GTest::Main
        GIFT
    )
    add_test("${TEST_NAME}" TEST_NAME)
endforeach()