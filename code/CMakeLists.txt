cmake_minimum_required(VERSION 3.13)
project(chemHMC  CXX)


# no need to install Eigen with cmake and use find package
#SET( EIGEN3_INCLUDE_DIR /home/marco/library/eigen )
#IF( NOT EIGEN3_INCLUDE_DIR )
#    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
#ENDIF()
#INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

#using the globally installed version of Eigen: sudo apt-get libeigen3-dev
# Eigen
#IF(NOT DEFINED EIGEN3_INCLUDE_DIR)
#  find_package(Eigen3 REQUIRED)
#ENDIF()
#INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

set(Kokkos_ENABLE_COMPILER_WARNINGS ON)
set(Kokkos_CXX_STANDARD 14)

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} -DDEBUG  -g   -lm  ")
  set(Kokkos_ENABLE_DEBUG ON)
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON)
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON)
  MESSAGE("DEBUG mode ON")
  MESSAGE("Kokkos_ENABLE_DEBUG = ${Kokkos_ENABLE_DEBUG}")
  MESSAGE("Kokkos_ENABLE_DEBUG_BOUNDS_CHECK = ${Kokkos_ENABLE_DEBUG_BOUNDS_CHECK}")
ENDIF()

MESSAGE("CMAKE_CXX_FLAGS   ${CMAKE_CXX_FLAGS}")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/kokkos" "external/kokkos")

add_subdirectory(include  include)



######################### git version ################
######################################################

include(cmake/CheckGit.cmake)
CheckGitSetup()

######################################################
######################################################


macro(add_target_with_lib TARGETNAME)
  # Make the executable
  add_executable(${TARGETNAME} ${ARGN})
  target_link_libraries(${TARGETNAME} PUBLIC Kokkos::kokkos)
  target_link_libraries(${TARGETNAME} PUBLIC read_infile)
  #target_link_libraries(${TARGETNAME} PUBLIC extra_library)
  
  target_link_libraries(${TARGETNAME} PUBLIC git_version)

 
endmacro()

add_subdirectory(main)
add_subdirectory(test)

