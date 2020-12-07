cmake_minimum_required(VERSION 3.14)

# Add C++ example target:
add_executable(hwsim-ex hwsim-ex.cpp)
target_link_libraries(hwsim-ex PRIVATE hwsim)
target_compile_features(hwsim-ex PRIVATE cxx_std_17)

# C wrapper example:
add_executable(libhwsim-ex libhwsim-ex.cpp)
target_link_libraries(libhwsim-ex PRIVATE libhwsim)
target_compile_features(libhwsim-ex PRIVATE cxx_std_17)

if(HWSIM_INSTALL_EXAMPLES)
    # Install copy of hwsim library:
    install(TARGETS libhwsim DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
    # Install examples:
    install(
        TARGETS hwsim-ex libhwsim-ex
        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples
    )
endif()
