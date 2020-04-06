Installation instructions:
*Note: for using the C++ or C library, it is not necessary to follow the steps below. Instead you can just add the source files of this library to a folder
inside your project and include ´add_subdirectory(<FOLDER_NAME>)´ in your CMakeLists file. To link your application to the library, see below.*

1. Unpack or download dependencies (see readme inside dependencies folder).
2. Create a build directory
    mkdir build
    cd build
3. Create CMake project directory and install:
    cmake -DCMAKE_BUILD_TYPE=STRING:Release ..
    cmake --install . --config Release --prefix <INSTALL_DIR>

Usage:

* C++ project:
  Include headers and link to Clothoids and HDF5 library. Using CMake this can easily be done by ´target_link_libraries(myApp PRIVATE hwsim)´
* C library:
  Include C header and link to hwsim library. Using CMake this can be done by ´target_link_libraries(myApp PRIVATE libhwsim)´ (or libhwsim-static)
* MATLAB wrappers:
  Install project. The Matlab files can afterwards be found in the <INSTALL_DIR>/matlab directory
* Python package:
  Install project and run ´pip install hwsim´ from the <INSTALL_DIR>/python directory