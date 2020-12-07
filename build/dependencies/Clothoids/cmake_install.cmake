# Install script for directory: D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win_.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY OPTIONAL MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Debug/Clothoids_win_.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win_.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY OPTIONAL MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Release/Clothoids_win_.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win_.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY OPTIONAL MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/MinSizeRel/Clothoids_win_.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win_.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY OPTIONAL MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/RelWithDebInfo/Clothoids_win_.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin/Clothoids_win_.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Debug/Clothoids_win_.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin/Clothoids_win_.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Release/Clothoids_win_.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin/Clothoids_win_.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/MinSizeRel/Clothoids_win_.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin/Clothoids_win_.dll")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/bin" TYPE SHARED_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/RelWithDebInfo/Clothoids_win_.dll")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win__static.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Debug/Clothoids_win__static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win__static.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/Release/Clothoids_win__static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win__static.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/MinSizeRel/Clothoids_win__static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib/Clothoids_win__static.lib")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/RelWithDebInfo/Clothoids_win__static.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/AABBtree.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/BaseCurve_using.hxx;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Biarc.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/BiarcList.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Circle.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Clothoid.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/ClothoidAsyPlot.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/ClothoidList.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Fresnel.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/G2lib.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Line.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/PolyLine.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/Triangle2D.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/PolynomialRoots-Utils.hh;D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include/PolynomialRoots.hh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/lib/include" TYPE FILE MESSAGE_NEVER FILES
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/AABBtree.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/BaseCurve_using.hxx"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Biarc.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/BiarcList.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Circle.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Clothoid.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/ClothoidAsyPlot.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/ClothoidList.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Fresnel.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/G2lib.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Line.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/PolyLine.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/src/Triangle2D.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/submodules/quarticRootsFlocke/src/PolynomialRoots-Utils.hh"
    "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/source/submodules/quarticRootsFlocke/src/PolynomialRoots.hh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "D:/Documents/Projects/hwsim/build/dependencies/Clothoids/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
