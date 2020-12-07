# Install script for directory: D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src

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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xheadersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/hdf5.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5api_adpt.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5public.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Apublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5ACpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Cpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Dpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Epubgen.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Epublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5ESpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Fpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDcore.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDdirect.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDfamily.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDhdfs.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDlog.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDmpi.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDmpio.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDmulti.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDros3.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDs3comms.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDsec2.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDstdio.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5FDwindows.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Gpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Ipublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Lpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Mpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5MMpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Opublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Ppublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5PLextern.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5PLpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Rpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Spublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Tpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5VLconnector.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5VLconnector_passthru.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5VLnative.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5VLpassthru.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5VLpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Zpublic.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5Epubgen.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5version.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/source/src/H5overflow.h"
    "D:/Documents/Projects/hwsim/build/dependencies/hdf5/H5pubconf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE OPTIONAL FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Debug/.pdb")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE FILE OPTIONAL FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/RelWithDebInfo/.pdb")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Debug/libhdf5_D.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/Release/libhdf5.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/MinSizeRel/libhdf5.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/bin/RelWithDebInfo/libhdf5.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibrariesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "D:/Documents/Projects/hwsim/build/dependencies/hdf5/CMakeFiles/hdf5-1.12.0.pc")
endif()

