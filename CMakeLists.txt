cmake_minimum_required(VERSION 3.18)
project(DCNN LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# defaults (might change)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_TBB "Enable Intel TBB support" OFF)
option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)

# default to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# compiler flags
if(MSVC)
    # MSVC flags for Release and Debug
    set(CMAKE_CXX_FLAGS_RELEASE "/O3 /Oi /Oy /GL /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    add_compile_options(/W4)
    # Disable Windows min/max macros
    add_compile_definitions(NOMINMAX)
    # Prevent winsock.h from being included
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    # Define SIMD macros for MSVC
    add_compile_definitions(__AVX2__ __SSE2__)
    
else()
    # GCC/Clang flags for Release and Debug
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -flto=auto -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -flto=auto")
    add_compile_options(-Wpedantic -Wall)
endif()


if(MSVC)
    add_compile_options(/W4)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wpedantic -Wall)
endif()

# debug build type
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(WIN32)
    if(MSVC)
        message("USING MSVC")
        set(OpenMP_C_FLAGS "/openmp:llvm")
        set(OpenMP_CXX_FLAGS "/openmp:llvm")
        set(OpenMP_C_LIB_NAMES "")
        set(OpenMP_CXX_LIB_NAMES "")
        set(OpenMP_C_LIBRARIES "")
        set(OpenMP_CXX_LIBRARIES "")
    elseif(MINGW)
        message("USING MINGW")
        set(OpenMP_C_FLAGS "-fopenmp")
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_C_LIB_NAMES "gomp;pthread")
        set(OpenMP_CXX_LIB_NAMES "gomp;pthread")
    endif()
endif()

# find packages
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

# output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# libraries
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/include
)

# template libraries instantiations
add_subdirectory(src) 

# Function to create executables
function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    if(ENABLE_CUDA)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    endif()
    
    target_link_libraries(${target_name} PRIVATE dcnn_lib)
endfunction()

add_subdirectory(examples)