cmake_minimum_required(VERSION 3.18)

project(DCNN LANGUAGES CXX)

# Add cmake modules directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(NASM_DIR "" CACHE PATH "Path to NASM executable") # User can set this to their NASM path

# boilerplate to enable assembly because NASM is not found by default
if(WIN32 AND MSVC)
    find_program(NASM_EXECUTABLE nasm
        HINTS
            ${NASM_DIR}
        PATHS
            "$ENV{PROGRAMFILES}/NASM"
            "$ENV{PROGRAMFILES(X86)}/NASM"
            "C:/NASM"
        ENV PATH
    )
    if(NASM_EXECUTABLE)
        message(STATUS "Found NASM: ${NASM_EXECUTABLE}")
        enable_language(ASM_NASM)
        set(CMAKE_ASM_NASM_COMPILER ${NASM_EXECUTABLE})
        set(CMAKE_ASM_NASM_OBJECT_FORMAT win64)
    endif()
elseif(LINUX)
    enable_language(ASM)
else()
    message("Unknown distribution, enabling assembly anyways")
    enable_language(ASM)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
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
    message(STATUS "Finding OpenMP")
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_TBB)
    message(STATUS "Finding TBB")
    find_package(TBB CONFIG REQUIRED)
    add_compile_definitions(USE_TBB)
endif()

if(ENABLE_MKL)
    include(FindMKL)
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "-arch=sm_89 --compiler-options -fPIC")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
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
)

# Template libraries instantiations
add_subdirectory(src)

# Utility functions
include(Utils)

add_subdirectory(examples)
add_subdirectory(benchmarks)