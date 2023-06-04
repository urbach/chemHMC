cmake_minimum_required(VERSION 3.13)
project(chemHMC  CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

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

#set(Kokkos_ENABLE_COMPILER_WARNINGS ON)
# set(Kokkos_CXX_STANDARD 14)

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS  " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} -DDEBUG  -g   -lm  ")
  # The following need to be set from the command line
  # set(Kokkos_ENABLE_DEBUG ON)
  # set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON)
  # set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON)
  MESSAGE("DEBUG mode ON")
  MESSAGE("Kokkos_ENABLE_DEBUG = ${Kokkos_ENABLE_DEBUG}")
  MESSAGE("Kokkos_ENABLE_DEBUG_BOUNDS_CHECK = ${Kokkos_ENABLE_DEBUG_BOUNDS_CHECK}")
ENDIF()

MESSAGE("CMAKE_CXX_FLAGS   ${CMAKE_CXX_FLAGS}")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/kokkos" "external/kokkos")
include_directories(${Kokkos_INCLUDE_DIRS_RET})

######################### git version ################
######################################################

include(cmake/CheckGit.cmake)
CheckGitSetup()

######################################################
###################### yaml-cpp ######################

find_package(yaml-cpp)

include_directories(${YAML_CPP_INCLUDE_DIR})

MESSAGE("YAML INCLUDE_DIR: ${YAML_CPP_INCLUDE_DIR}")
MESSAGE("YAML LIBRARIES: ${YAML_CPP_LIBRARIES}")
MESSAGE("YAML LIBRARY_DIR: ${YAML_CPP_LIBRARY_DIR}")
IF(NOT yaml-cpp_FOUND)
    MESSAGE("find_package faild to find yaml-cpp: you need to do it manually")
    MESSAGE("hint:  yaml-cpp_DIR or CMAKE_PREFIX_PATH need to be global path or relative to the source dir")
    MESSAGE(FATAL_ERROR)
ENDIF()

######################################################
###################### fmt: formating library ######################

find_package(fmt)

IF(NOT fmt_FOUND)
    MESSAGE("find_package faild to find fmt: you need to do it manually")
    MESSAGE("hint:  fmt_DIR or CMAKE_PREFIX_PATH need to be global path or relative to the source dir")
    MESSAGE(FATAL_ERROR)
ENDIF()

######################################################
######################################################

add_subdirectory(include  include)

#######################################################

target_include_directories(chemHMC 
PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
) 
#######################################################

macro(add_target_with_lib TARGETNAME)
  # Make the executable
  add_executable(${TARGETNAME} ${ARGN})
  target_link_libraries(${TARGETNAME} PUBLIC Kokkos::kokkos)
  target_link_libraries(${TARGETNAME} PUBLIC chemHMC)
  target_link_libraries(${TARGETNAME} PUBLIC git_version)
endmacro()

add_subdirectory(main)
add_subdirectory(test)
