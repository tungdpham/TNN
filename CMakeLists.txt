cmake_minimum_required(VERSION 3.18)

project(TNN LANGUAGES CXX)

# Add cmake modules directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(NASM_DIR "" CACHE PATH "Path to NASM executable") # User can set this to their NASM path

include(FindNASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_CUDNN "Enable cuDNN support (auto-detected if not specified)" ON)
option(ENABLE_TBB "Enable Intel TBB support" ON)
option(ENABLE_MKL "Enable Intel MKL support" OFF)
option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)

# default to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# debug build type
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include(Dependencies)

# Find packages
if(ENABLE_OPENMP)
    include(FindOpenMP)
endif()

if(ENABLE_TBB)
    include(FindTBB)
endif()

if(ENABLE_MKL)
    include(FindMKL)
endif()

if(ENABLE_CUDA)
    include(FindCUDA)
endif()

# Compiler flags
include(CompilerFlags)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${asio_SOURCE_DIR}/asio/include
    ${stb_SOURCE_DIR}
)

# Linkers
include(LinkTBB)
include(LinkMKL)
include(LinkCUDA)
include(LinkWindowsLibs)
include(LinkOpenMP)

# Template libraries instantiations
add_subdirectory(src)

# Utility functions
include(ExecutableFunctions)

add_subdirectory(examples)
add_subdirectory(benchmarks)

# Enable testing
enable_testing()
add_subdirectory(unit_tests)