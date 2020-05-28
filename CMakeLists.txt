cmake_minimum_required(VERSION 3.14)
project(hwsim VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

# PRIVATE = only required to build current target
# INTERFACE = only required to link against current target
# PUBLIC = both required to build current target AND link against it

# Options:
option(HWSIM_COMPAT "Build with compatibility mode." OFF)
option(HWSIM_INSTALL_LIB "Install the C library." ON)
option(HWSIM_INSTALL_PYTHON "Install the python wrapper." ON)
option(HWSIM_INSTALL_MATLAB "Install the MATLAB wrapper." OFF)
option(HWSIM_INSTALL_EXAMPLES "Install the example executables." ON)

message(STATUS "Building in compatibility mode? ${HWSIM_COMPAT}")
message(STATUS "Installing C library? ${HWSIM_INSTALL_LIB}")
message(STATUS "Installing python wrapper? ${HWSIM_INSTALL_PYTHON}")
message(STATUS "Installing MATLAB wrapper? ${HWSIM_INSTALL_MATLAB}")
message(STATUS "Installing example executables? ${HWSIM_INSTALL_EXAMPLES}")

# Display all warnings if in debug mode:
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(MSVC)
    add_compile_options(/W3) # W4 produced unreferenced/unused parameter warnings
  else()
    add_compile_options(-Wall -Wextra -pedantic)
  endif()
  set(DEBUG_CONFIG ON)
else()
  set(DEBUG_CONFIG OFF)
endif()

# Include (and build if necessary) our dependency targets:
add_subdirectory(dependencies)

# Add a C++ interface library:
add_library(hwsim INTERFACE)
target_include_directories(hwsim
  INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(hwsim
  INTERFACE
    Clothoids
    HDF5
)
target_compile_features(hwsim INTERFACE cxx_std_17)
if(HWSIM_COMPAT)
  target_compile_definitions(hwsim INTERFACE COMPAT)
endif()


# Add C library wrapper:
add_library(libhwsim SHARED C/hwsim.cpp)
add_library(libhwsim-static STATIC C/hwsim.cpp)
target_compile_definitions(libhwsim PRIVATE HWSIM_SHARED)
target_include_directories(libhwsim
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/C>
)
target_include_directories(libhwsim-static
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/C>
)
target_link_libraries(libhwsim PRIVATE hwsim)
target_link_libraries(libhwsim-static PRIVATE hwsim)
set_target_properties(libhwsim PROPERTIES PREFIX "lib" LIBRARY_OUTPUT_NAME "hwsim" RUNTIME_OUTPUT_NAME "hwsim")
set_target_properties(libhwsim-static PROPERTIES PREFIX "lib" ARCHIVE_OUTPUT_NAME "hwsim-static")

# Add and install Matlab mex wrappers:
if(HWSIM_INSTALL_MATLAB)
  add_subdirectory(matlab)
endif()

# Configure python wrapper:
file(GENERATE
  OUTPUT python/hwsim/hwsim/_wrapper.py
  INPUT python/hwsim/hwsim/_wrapper.py.in
)

# Tests:
add_subdirectory(tests)

# Examples:
add_subdirectory(examples)

# Install targets:
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include) # C++ headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/ DESTINATION ${CMAKE_INSTALL_PREFIX}/scenarios) # Default scenarios
if(HWSIM_INSTALL_LIB)
  install(
    TARGETS libhwsim libhwsim-static
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  )
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/C/hwsim.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(HWSIM_INSTALL_PYTHON)
  # Install static files
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python
    FILES_MATCHING
    PATTERN "*"
    PATTERN "*.in" EXCLUDE
  )
  # Install generated files
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/ DESTINATION ${CMAKE_INSTALL_PREFIX}/python)
  # Install shared library
  install(TARGETS libhwsim DESTINATION ${CMAKE_INSTALL_PREFIX}/python/hwsim/hwsim)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
