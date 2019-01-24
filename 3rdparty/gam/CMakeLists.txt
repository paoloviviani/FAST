cmake_minimum_required(VERSION 3.8)

# workaround for setting policies before 3.13
if(${CMAKE_VERSION} VERSION_LESS 3.13)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.13)
endif()

# declare the project
project(GAM VERSION 0.1.0
            LANGUAGES CXX)

# add path for custom CMake modules
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check/set build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE AND
    NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL)$")
  message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# No in-tree build allowed.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source build are not allowed.
Please create a directory directory and run cmake from there, passing the path
to this source directory as the last argumente.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()

# Build options:
option(GAM_ENABLE_DOXYGEN "Use doxygen to generate the shad API documentation" OFF)
option(GAM_ENABLE_UNIT_TEST "Enable the compilation of Unit Tests" ON)

# check/set runtime system
set(
  GAM_RUNTIME_SYSTEM "libfabric" CACHE STRING
  "Runtime system to be used as backend (Default=libfabric, Supported=libfabric)")

# check runtime system
if (GAM_RUNTIME_SYSTEM AND NOT GAM_RUNTIME_SYSTEM  MATCHES
    "^(libfabric)$")
  message(FATAL_ERROR "Invalid value for GAM_RUNTIME_SYSTEM: ${GAM_RUNTIME_SYSTEM}")
endif()

# define the target library
add_library(gam INTERFACE)
add_library(gam::gam ALIAS gam)

# target attributes: compile features
target_compile_features(gam INTERFACE cxx_std_11)

# target attributes: include directories
target_include_directories(gam INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)

# target attributes: dependencies
find_package(Threads REQUIRED)
target_link_libraries(gam INTERFACE Threads::Threads)

# target attributes: implementation-specific dependencies
if (GAM_RUNTIME_SYSTEM STREQUAL "libfabric")
  message(STATUS "Using libfabric runtime system.")
  find_package(Libfabric REQUIRED)
  target_link_libraries(gam INTERFACE libfabric::libfabric)
endif()

# Unit tests
if (GAM_ENABLE_UNIT_TEST)
  include(CTest)
  add_subdirectory(tests)
endif()

# TODO Doxygen

# clang-format
include(CheckClangTools)
if(CLANG_FORMAT_EXE)
  # Additional targets to perform clang-format/clang-tidy
  # Get all project files
  file(GLOB_RECURSE
    ALL_CXX_SOURCE_FILES
    *.[chi]pp *.[chi]xx *.cc *.hh *.ii *.[CHI] *.[ch]
    )

  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT_EXE}
    -i
    ${ALL_CXX_SOURCE_FILES}
    )
endif()
