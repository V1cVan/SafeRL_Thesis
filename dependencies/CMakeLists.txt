# ----- CLOTHOIDS LIBRARY -----
if(DEBUG_CONFIG)
  set(DEBUG_SUFFIX "_debug")
endif()

# Determine name of Clothoids library (copied from the CMakeLists.txt file of the Clothoids project)
IF ( APPLE )
  SET( CLOTHOIDS_LIB Clothoids_osx_static${DEBUG_SUFFIX} )
ELSEIF( LINUX OR UNIX )
  SET( CLOTHOIDS_LIB Clothoids_linux_static${DEBUG_SUFFIX} )
ELSE()
  SET( CLOTHOIDS_LIB Clothoids_win_${BITS}_static${DEBUG_SUFFIX} )
ENDIF()
# Find library and build from source if not found
set(CLOTHOIDS_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/Clothoids.tar.gz)
set(CLOTHOIDS_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/Clothoids/source)
if(NOT EXISTS ${CLOTHOIDS_SOURCE_DIR})
  message(STATUS "No source files for the Clothoids library found. Extracting archive ${CLOTHOIDS_ARCHIVE}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/Clothoids)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CLOTHOIDS_ARCHIVE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Clothoids
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/Clothoids/Clothoids ${CLOTHOIDS_SOURCE_DIR})
  if(NOT EXISTS ${CLOTHOIDS_SOURCE_DIR})
    message(FATAL_ERROR "Error while extracting source files for the Clothoids library :(")
  endif()
endif()
if(DEBUG_CONFIG)
  # In debug configuration, set Clothoids' install directory to special debug folder
  # This is to be able to differentiate between the debug and release binaries
  # (If installed in the same directory find_library returns the wrong one)
  set(CLOTHOIDS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/Clothoids/install-debug)
else()
  set(CLOTHOIDS_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/Clothoids/install)
endif()
message(STATUS "Looking for ${CLOTHOIDS_LIB} in ${CLOTHOIDS_INSTALL_DIR}/lib")
unset(ClothoidsLib CACHE) # Force find_library every time (otherwise cached library is used)
find_library(ClothoidsLib NAMES ${CLOTHOIDS_LIB} PATHS ${CLOTHOIDS_INSTALL_DIR}/lib NO_DEFAULT_PATH)
if(NOT ClothoidsLib)
  message(STATUS "Clothoids library not found, building from source...")

  # Build Clothoids library
  set(CLOTHOIDS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Clothoids)
  # Create project directory inside current build directory and set proper build configuration
  execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} -DBUILD_SHARED:BOOL=ON -S ${CLOTHOIDS_SOURCE_DIR} -B ${CLOTHOIDS_BINARY_DIR})
  # Build the Clothoids project
  execute_process(COMMAND ${CMAKE_COMMAND} --build ${CLOTHOIDS_BINARY_DIR} --config ${CMAKE_BUILD_TYPE})
  # And install it (creates lib directory and moves header files and built libraries)
  execute_process(COMMAND ${CMAKE_COMMAND} --install ${CLOTHOIDS_BINARY_DIR} --config ${CMAKE_BUILD_TYPE} --prefix ${CLOTHOIDS_INSTALL_DIR})
  # Because the Clothoids library is always installed in the same location, move it manually:
  execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${CLOTHOIDS_SOURCE_DIR}/lib ${CLOTHOIDS_INSTALL_DIR})
  # file(COPY ${CLOTHOIDS_SOURCE_DIR}/lib/ DESTINATION ${CLOTHOIDS_INSTALL_DIR})
  # file(REMOVE_RECURSE ${CLOTHOIDS_SOURCE_DIR}/lib)

  # Once it is installed, try to find the library again:
  find_library(ClothoidsLib NAMES ${CLOTHOIDS_LIB} PATHS ${CLOTHOIDS_INSTALL_DIR}/lib NO_DEFAULT_PATH)

  # Safety check:
  if(NOT ClothoidsLib)
    message(FATAL_ERROR "Clothoids library could not be found after building from source :(")
  endif()
endif()
message(STATUS "Using Clothoids library: ${ClothoidsLib}")
# Add an interface library to the Clothoids library:
add_library(Clothoids INTERFACE)
target_include_directories(Clothoids
  INTERFACE
    $<INSTALL_INTERFACE:dependencies/Clothoids/include>
    $<BUILD_INTERFACE:${CLOTHOIDS_INSTALL_DIR}/include>
)
target_link_libraries(Clothoids INTERFACE ${ClothoidsLib})
# If the Matlab wrapper for the hwsim library is required, install
# extra mex targets for the Clothoids library:
if(HWSIM_INSTALL_MATLAB)
  # Find Matlab package:
  find_package(Matlab)
  # Add mex targets for Clothoids library:
  set(CLOTHOIDS_MEX_SRC ${CLOTHOIDS_SOURCE_DIR}/src_mex)
  add_library(ClothoidsMex INTERFACE)
  target_include_directories(ClothoidsMex INTERFACE ${CLOTHOIDS_MEX_SRC})
  matlab_add_mex(
    NAME ClothoidCurveMexWrapper MODULE
    SRC ${CLOTHOIDS_MEX_SRC}/mex_ClothoidCurveMexWrapper.cc
    LINK_TO Clothoids ClothoidsMex
    R2017b
  )
  matlab_add_mex(
    NAME ClothoidListMexWrapper MODULE
    SRC ${CLOTHOIDS_MEX_SRC}/mex_ClothoidListMexWrapper.cc
    LINK_TO Clothoids ClothoidsMex
    R2017b
  )
  matlab_add_mex(
    NAME ClothoidSplineG2MexWrapper MODULE
    SRC ${CLOTHOIDS_MEX_SRC}/mex_ClothoidSplineG2MexWrapper.cc
    LINK_TO Clothoids ClothoidsMex
    R2017b
  )

  # Install Clothoids mex targets:
  install(
    TARGETS ClothoidCurveMexWrapper ClothoidListMexWrapper ClothoidSplineG2MexWrapper
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dependencies/Clothoids/matlab
  )
  install(
    FILES
      ${CLOTHOIDS_SOURCE_DIR}/matlab/ClothoidCurve.m
      ${CLOTHOIDS_SOURCE_DIR}/matlab/ClothoidList.m
      ${CLOTHOIDS_SOURCE_DIR}/matlab/ClothoidSplineG2.m
      ${CLOTHOIDS_SOURCE_DIR}/matlab/CurveBase.m
    DESTINATION ${CMAKE_INSTALL_PREFIX}/dependencies/Clothoids/matlab
  )
endif()
# Install Clothoids library in dependencies folder:
install(DIRECTORY ${CLOTHOIDS_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX}/dependencies/Clothoids)


# ----- HDF5 library -----
# Find HDF5 library and build from source if not found
set(HDF5_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/hdf5-1.12.0.tar.gz)
set(HDF5_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5/source)
if(NOT EXISTS ${HDF5_SOURCE_DIR})
  message(STATUS "No source files for the HDF5 library found. Extracting archive ${HDF5_ARCHIVE}")
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/hdf5)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${HDF5_ARCHIVE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hdf5
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/hdf5/hdf5-1.12.0 ${HDF5_SOURCE_DIR})
  if(NOT EXISTS ${HDF5_SOURCE_DIR})
    message(FATAL_ERROR "Error while extracting source files for the HDF5 library :(")
  endif()
endif()
if(DEBUG_CONFIG)
  # In debug configuration, set HDF5 install directory to special debug folder
  # This is to be able to differentiate between the debug and release binaries
  # (If installed in the same directory find_library returns the wrong one)
  set(HDF5_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5/install-debug)
else()
  set(HDF5_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5/install)
endif()
message(STATUS "Looking for hdf5 in ${HDF5_INSTALL_DIR}/cmake/hdf5 and ${HDF5_INSTALL_DIR}/share/cmake/hdf5")
unset(HDF5Package CACHE) # Force find_package every time (otherwise cached package is used)
unset(HDF5Package_FOUND CACHE)
unset(HDF5Package_DIR CACHE)
find_package(HDF5Package NAMES hdf5 COMPONENTS C HL static NO_PACKAGE_ROOT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH) # First look if a system-wide hdf5 package is available
find_package(HDF5Package NAMES hdf5 COMPONENTS C HL static PATHS ${HDF5_INSTALL_DIR}/cmake/hdf5 ${HDF5_INSTALL_DIR}/share/cmake/hdf5 NO_DEFAULT_PATH)
if(NOT HDF5Package_FOUND)
  message(STATUS "HDF5 library not found, building from source...")

  # Build HDF5 library
  set(HDF5_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hdf5)
  # Create project directory inside current build directory and set proper build configuration
  execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS:BOOL=OFF -DBUILD_TESTING:BOOL=OFF -DHDF5_BUILD_TOOLS:BOOL=OFF -DHDF5_BUILD_EXAMPLES:BOOL=OFF -DHDF5_BUILD_CPP_LIB:BOOL=OFF -DINSTALLDIR:STRING=${HDF5_SOURCE_DIR} -S ${HDF5_SOURCE_DIR} -B ${HDF5_BINARY_DIR})
  # Build the HDF5 project
  execute_process(COMMAND ${CMAKE_COMMAND} --build ${HDF5_BINARY_DIR} --config ${CMAKE_BUILD_TYPE})
  # And install it (creates the bin, include, lib and cmake folder)
  execute_process(COMMAND ${CMAKE_COMMAND} --install ${HDF5_BINARY_DIR} --config ${CMAKE_BUILD_TYPE} --prefix ${HDF5_INSTALL_DIR})

  # Once it is installed, try to find the library again:
  find_package(HDF5Package NAMES hdf5 COMPONENTS C HL static PATHS ${HDF5_INSTALL_DIR}/cmake/hdf5 ${HDF5_INSTALL_DIR}/share/cmake/hdf5 NO_DEFAULT_PATH)

  # Safety check:
  if(NOT HDF5Package_FOUND)
    message(FATAL_ERROR "HDF5 library could not be found after building from source :(")
  endif()
endif()
message(STATUS "Using HDF5 package: ${HDF5Package_DIR}")
# The HDF5 package will export a target to the static C library: 'hdf5-static'
# (other targets can be found in the targets cmake file located in HDF5_DIR).
# Add an interface library to the HDF5 library:
add_library(HDF5 INTERFACE)
target_include_directories(HDF5 INTERFACE ${HDF5_INCLUDE_DIR})
target_link_libraries(HDF5 INTERFACE hdf5-static debug hdf5_hl-static)