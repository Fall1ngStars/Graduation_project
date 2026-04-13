#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ranger_msgs::ranger_msgs__rosidl_generator_py" for configuration ""
set_property(TARGET ranger_msgs::ranger_msgs__rosidl_generator_py APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ranger_msgs::ranger_msgs__rosidl_generator_py PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libranger_msgs__rosidl_generator_py.so"
  IMPORTED_SONAME_NOCONFIG "libranger_msgs__rosidl_generator_py.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ranger_msgs::ranger_msgs__rosidl_generator_py )
list(APPEND _IMPORT_CHECK_FILES_FOR_ranger_msgs::ranger_msgs__rosidl_generator_py "${_IMPORT_PREFIX}/lib/libranger_msgs__rosidl_generator_py.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
