#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hdf5-static" for configuration "Debug"
set_property(TARGET hdf5-static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(hdf5-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libhdf5_D.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5-static "${_IMPORT_PREFIX}/lib/libhdf5_D.lib" )

# Import target "hdf5_hl-static" for configuration "Debug"
set_property(TARGET hdf5_hl-static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(hdf5_hl-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libhdf5_hl_D.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hdf5_hl-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hdf5_hl-static "${_IMPORT_PREFIX}/lib/libhdf5_hl_D.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
