# Install script for directory: D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhlheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5DOpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5DSpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5IMpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5LTpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5PTpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5TBpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/H5LDpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/hl/src/hdf5_hl.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhllibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE OPTIONAL FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Debug/.pdb")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE OPTIONAL FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/RelWithDebInfo/.pdb")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhllibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Debug/libhdf5_hl_D.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Release/libhdf5_hl.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/MinSizeRel/libhdf5_hl.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/RelWithDebInfo/libhdf5_hl.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xhllibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/hdf5_hl-1.12.0.pc")
endif()

