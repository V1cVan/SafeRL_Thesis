# Install script for directory: D:/Documents/Projects/hwsim/build/dependencies/hdf5/source

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/HDF_Group/HDF5/1.12.0")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xconfiginstallx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/cmake/hdf5/hdf5-targets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/cmake/hdf5/hdf5-targets.cmake"
         "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/cmake/hdf5/hdf5-targets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/cmake/hdf5/hdf5-targets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets-debug.cmake")
  endif()
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets-minsizerel.cmake")
  endif()
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets-relwithdebinfo.cmake")
  endif()
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/Export/cmake/hdf5/hdf5-targets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xconfiginstallx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/hdf5-config.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xconfiginstallx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake/hdf5" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/hdf5-config-version.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/libhdf5.settings")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhdfdocumentsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/COPYING")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhdfdocumentsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/release_docs/USING_HDF5_CMake.txt"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/release_docs/COPYING"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/release_docs/RELEASE.txt"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/release_docs/USING_HDF5_VS.txt"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/msvcp140.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/msvcp140_1.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/msvcp140_2.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/msvcp140_codecvt_ids.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/vcruntime140_1.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/vcruntime140.dll"
    "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Redist/MSVC/14.27.29016/x64/Microsoft.VC142.CRT/concrt140.dll"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE DIRECTORY FILES "")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/Documents/Projects/hwsim/build/dependencies/hdf5/src/cmake_install.cmake")
  include("D:/Documents/Projects/hwsim/build/dependencies/hdf5/hl/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/Documents/Projects/hwsim/build/dependencies/hdf5/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
