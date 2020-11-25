cmake_minimum_required(VERSION 3.14)

include(CTest)
enable_testing()

# Add Clothoids test:
add_executable(Clothoids-test Clothoids-test.cpp)
target_link_libraries(Clothoids-test PRIVATE hwsim)

# Add Eigen test:
add_executable(Eigen-test Eigen-test.cpp)
target_link_libraries(Eigen-test PRIVATE Eigen)

# Add C++ test target:
add_executable(hwsim-test hwsim-test.cpp)
target_link_libraries(hwsim-test PRIVATE hwsim doctest)
target_compile_features(hwsim-test PRIVATE cxx_std_17)

# C wrapper test:
add_executable(libhwsim-test libhwsim-test.cpp)
target_link_libraries(libhwsim-test PRIVATE libhwsim doctest)
target_compile_features(libhwsim-test PRIVATE cxx_std_17)

# Performance test:
add_executable(perf-test perf-test.cpp)
target_link_libraries(perf-test PRIVATE hwsim)

# Add tests to the cmake unit tests:
add_test(NAME hwsimTest COMMAND hwsim-test WORKING_DIRECTORY ${hwsim_BINARY_DIR})
add_test(NAME libhwsimTest COMMAND libhwsim-test WORKING_DIRECTORY ${hwsim_BINARY_DIR})
