cmake_minimum_required(VERSION 3.18)
project(DCNN LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Defaults
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_TBB "Enable Intel TBB support" OFF)
option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -flto=auto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -flto=auto")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

# Debug override
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Find packages
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_TBB)
    find_package(TBB REQUIRED)
    add_compile_definitions(USE_TBB)
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

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include
)

add_library(dcnn_instantiations STATIC src/instantiations.cpp)
target_link_libraries(dcnn_instantiations PUBLIC m)

add_library(dcnn_lib INTERFACE)

target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations m)

if(ENABLE_OPENMP)
    target_link_libraries(dcnn_instantiations PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(dcnn_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    target_link_libraries(dcnn_instantiations PUBLIC TBB::tbb)
    target_link_libraries(dcnn_lib INTERFACE TBB::tbb)
endif()

if(ENABLE_CUDA)
    target_link_libraries(dcnn_instantiations PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    target_link_libraries(dcnn_lib INTERFACE CUDA::cudart CUDA::cublas CUDA::curand)
endif()

# Function to create executables
function(create_executable target_name source_file)
    if(ENABLE_CUDA)
        add_executable(${target_name} ${source_file})
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    else()
        add_executable(${target_name} ${source_file})
    endif()
    
    target_link_libraries(${target_name} PRIVATE dcnn_lib)
    
    # if(${target_name} MATCHES "network|distributed|pipeline")
    #     target_link_libraries(${target_name} PRIVATE pthread)
    # endif()
endfunction()

add_subdirectory(examples)
