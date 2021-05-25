CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(tdm_sparse_coding)



############## Version
SET(TDM_SPARSE_CODING_VERSION_MAJOR 1)
SET(TDM_SPARSE_CODING_VERSION_MINOR 0)
SET(TDM_SPARSE_CODING_VERSION_PATCH 0)
CONFIGURE_FILE(config.h.in config.h)

############## Required Libraries

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules" )

find_package(OpenMP     REQUIRED)
find_package(SL         REQUIRED)
find_package(Eigen3     REQUIRED)

if (OPENMP_FOUND)
    MESSAGE("OpenMP Library: Found!" )
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -std=c++17 -lstdc++fs")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17 -lstdc++fs")
else (!OPENMP_FOUND)
    MESSAGE( WARNING "OpenMP Library: Not found!" )
endif()

############## Default target to build

if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

MESSAGE("Compiled in ${CMAKE_BUILD_TYPE}")

SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)
MESSAGE("Export compile commands: ${CMAKE_EXPORT_COMPILE_COMMANDS}")

############## Output to the "bin" and "lib" folders

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin )
SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY ./lib )
SET( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ./lib )
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
	STRING( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ./bin )
	SET( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ./lib )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

######## Includes for everybody ############

INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/src" )
INCLUDE_DIRECTORIES( ${SL_INCLUDE_DIR} )
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )

######## Install directories ############

SET( INSTALL_INCLUDE_DIR /usr/local/include/vic/tdm_sparse_coding )
SET( INSTALL_LIB_DIR /usr/local/lib64 )
SET( INSTALL_BIN_DIR /usr/local/bin )

MESSAGE("Install include dir: ${INSTALL_INCLUDE_DIR}")
MESSAGE("Install lib dir: ${INSTALL_LIB_DIR}")
MESSAGE("Install bin dir: ${INSTALL_BIN_DIR}")

######## Uninstall ##########
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall-cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

########################################################
##### TDM_SPARSE_CODING LIBRARY ########################
########################################################

ADD_LIBRARY( vic_tdm_sparse_coding 
             src/vic/tdm_sparse_coding/dictionary_coder.hpp
             src/vic/tdm_sparse_coding/dictionary_coder.cpp
             src/vic/tdm_sparse_coding/dictionary_coder_ksvd.cpp
             src/vic/tdm_sparse_coding/dictionary_coder_ksvd.hpp
             src/vic/tdm_sparse_coding/error_evaluator.hpp
             src/vic/tdm_sparse_coding/error_evaluator.cpp
             src/vic/tdm_sparse_coding/incremental_sparse_coder.hpp
             src/vic/tdm_sparse_coding/incremental_sparse_coder.cpp
             src/vic/tdm_sparse_coding/incremental_sparse_coder_omp.hpp
             src/vic/tdm_sparse_coding/incremental_sparse_coder_omp.cpp
             src/vic/tdm_sparse_coding/matching_pursuit.hpp
             src/vic/tdm_sparse_coding/matching_pursuit.cpp
             src/vic/tdm_sparse_coding/signal_stream.hpp
             src/vic/tdm_sparse_coding/signal_stream.cpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder.hpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder.cpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder_reservoir.hpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder_reservoir.cpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder_uniform.hpp
             src/vic/tdm_sparse_coding/streaming_coreset_builder_uniform.cpp
             src/vic/tdm_sparse_coding/dictionary_coding_budget_optimizer.hpp
             src/vic/tdm_sparse_coding/dictionary_coding_budget_optimizer.cpp
             src/vic/tdm_sparse_coding/dictionary_coding_budget_optimizer_greedy_grow.hpp
             src/vic/tdm_sparse_coding/dictionary_coding_budget_optimizer_greedy_grow.cpp
)

TARGET_LINK_LIBRARIES( vic_tdm_sparse_coding
     ${SL_LIBRARIES}
     ${EIGEN3_LIBRARIES} 
     ${OpenCV_LIBRARIES} 
)

file(GLOB config_files "src/vic/tdm_sparse_coding/*.hpp")
install (FILES ${config_files} DESTINATION ${INSTALL_INCLUDE_DIR}/base)
install (TARGETS vic_tdm_sparse_coding DESTINATION ${INSTALL_LIB_DIR})

#############################################################
##### TEST APPP  ############################################
#############################################################

ADD_EXECUTABLE (vic_tdm_test_volume_compression
               apps/test_volume_compression/float_xarray.cpp
               apps/test_volume_compression/float_xarray.hpp
               apps/test_volume_compression/ht4_volume_layout.hpp
               apps/test_volume_compression/raw_volume.cpp
               apps/test_volume_compression/raw_volume.hpp
               apps/test_volume_compression/volume_tile_stream.cpp
               apps/test_volume_compression/volume_tile_stream.hpp
               apps/test_volume_compression/test_volume_compression.cpp
     
)

TARGET_LINK_LIBRARIES( vic_tdm_test_volume_compression
     ${SL_LIBRARIES}
     vic_tdm_sparse_coding
)

install (TARGETS vic_tdm_test_volume_compression DESTINATION ${INSTALL_BIN_DIR})

TARGET_INCLUDE_DIRECTORIES(vic_tdm_test_volume_compression PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

