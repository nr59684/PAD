cmake_minimum_required(VERSION 3.12.0)
project( PAD_EXERCISES LANGUAGES CXX VERSION 1.0.0 )

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if( NOT CMAKE_BUILD_TYPE )       # be explicit, if you want a debug build
  set( CMAKE_BUILD_TYPE Release) # CMAKE_BUILD_TYPE information is also
endif()

add_subdirectory( range-v3 )

# Optionally include subfolder. Since exercises are independent,
# you can save compile time if you just build the required parts.
option( ASC_BUILD_EX01 "Builds PAD ex01" OFF )
if( ASC_BUILD_EX01 )
  add_subdirectory( ex01 )
endif( ASC_BUILD_EX01 )

option( ASC_BUILD_EX03 "Builds PAD ex03" OFF )
if( ASC_BUILD_EX03 )
  add_subdirectory( ex03 )
endif( ASC_BUILD_EX03 )

option( ASC_BUILD_EX04 "Builds PAD ex04" OFF )
if( ASC_BUILD_EX04 )
  add_subdirectory( ex04 )
endif( ASC_BUILD_EX04 )

option( ASC_BUILD_EX05 "Builds PAD ex05" ON )
if( ASC_BUILD_EX05 )
  add_subdirectory( ex05 )
endif( ASC_BUILD_EX05 )
