cmake_minimum_required(VERSION 3.14)

include(CTest)
enable_testing()

# Add Clothoids test:
add_executable(Clothoids-test Clothoids-test.cpp)
target_link_libraries(Clothoids-test PRIVATE hwsim)

# Add C++ test target:
add_executable(hwsim-test hwsim-test.cpp)
message(STATUS "hwsim_SOURCE_DIR=${hwsim_SOURCE_DIR}")
target_link_libraries(hwsim-test PRIVATE hwsim)
target_compile_features(hwsim-test PRIVATE cxx_std_17)

# C wrapper test:
add_executable(libhwsim-test libhwsim-test.cpp)
target_link_libraries(libhwsim-test PRIVATE libhwsim)
target_compile_features(libhwsim-test PRIVATE cxx_std_17)

if(HWSIM_INSTALL_TESTS OR DEBUG_CONFIG)
    # Install copy of hwsim library:
    install(TARGETS libhwsim DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)
    # Install tests:
    install(
        TARGETS hwsim-test libhwsim-test
        DESTINATION ${CMAKE_INSTALL_PREFIX}/tests
    )
endif()

# Add tests to the cmake unit tests:
#add_test(NAME hwsim-lib-test COMMAND hwsim-test)
