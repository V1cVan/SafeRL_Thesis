## Requirements:

* A compiler supporting C++17 static inline variables and `std::optional` features. Support for older compilers not supporting static inline variables, but supporting an experimental implementation of optionals through `std::experimental::optional` (e.g. GCC-6.0) can be enabled through the `COMPAT` flag.
* CMake 3.14 or higher (3.16.3 or higher for installing the Matlab library).

## Dependencies:

1. [Clothoids](https://www.github.com/ebertolazzi/Clothoids)
2. [HDF5](https://www.hdfgroup.org/downloads/hdf5/source-code/)

CMake will automatically extract the archives included in the dependencies folder, compile the sources and link with them.

*Note: the build process can be sped up considerably if the HDF5 library is already pre-installed on the system.*

## Installation instructions:
*Note: for using the C++ or C library, it is not necessary to follow the steps below. Instead you can just add the source files of this library to a folder
inside your project and include `add_subdirectory(<FOLDER_NAME>)` in your CMakeLists file. To link your application to the library, see below.*

1. Create a build directory

        mkdir build
        cd build
2. Create CMake project directory and install:

        cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> [options] ..
        cmake --build . --target install --config Release

    Possible options:

    * -DHWSIM_COMPAT:BOOL=[ON/OFF]

      Build with compatibility mode (to enable backward support for older compilers wrapping C++17 features inside an experimental module), default OFF (requires C++17 support of your compiler).

    * -DHWSIM_INSTALL_LIB:BOOL=[ON/OFF]
      
      Installs the C library and headers, default ON
    * -DHWSIM_INSTALL_MATLAB:BOOL=[ON/OFF]

      Installs the MATLAB wrappers, default OFF. **Note that this requires an active Matlab installation to be present on the system!**
    * -DHWSIM_INSTALL_PYTHON:BOOL=[ON/OFF]

      Installs the python wrappers, default ON.
    * -DHWSIM_INSTALL_TESTS:BOOL=[ON/OFF]

      Installs the test executables, default OFF.

## Usage:

* C++ project:
  Include headers and link to Clothoids and HDF5 library. Using CMake this can easily be done by `target_link_libraries(myApp PRIVATE hwsim)`
* C library:
  Include C header and link to hwsim library. Using CMake this can be done by `target_link_libraries(myApp PRIVATE libhwsim)` (or libhwsim-static)
* MATLAB wrappers:
  Install project with the `HWSIM_INSTALL_MATLAB` flag enabled. The Matlab files can afterwards be found in the <INSTALL_DIR>/matlab directory
* Python package:
  Install project and run `pip install hwsim` from the <INSTALL_DIR>/python directory

## Tested builds:

* Windows 10 using Visual Studio Build Tools 2019
* Linux using GCC 6.4.0 (with compatibility mode enabled)