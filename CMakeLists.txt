cmake_minimum_required(VERSION 3.18)
project(DCNN LANGUAGES CXX ASM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# defaults (might change)
option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_TBB "Enable Intel TBB support" ON)
option(ENABLE_DEBUG "Enable debug build with sanitizers" OFF)

# default to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# debug build type
if(ENABLE_DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(WIN32)
    
endif()

# find packages
find_package(Protobuf REQUIRED)

if(ENABLE_OPENMP)
    message("Finding OpenMP")
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_TBB)
    message("Finding TBB")
    find_package(TBB CONFIG REQUIRED)
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

# compiler flags
if(MSVC)
    message("Using MSVC")
    # MSVC flags for Release and Debug
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    add_compile_options(/W4)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    add_compile_definitions(__AVX2__ __SSE2__)
    if(OpenMP_CXX_FOUND)
        add_compile_options(/openmp:llvm)
    endif()
else()
    # GCC/Clang flags for Release and Debug
    if(MINGW)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native")
        add_compile_definitions(NOMINMAX)
        add_compile_definitions(WIN32_LEAN_AND_MEAN)
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -flto=auto -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -flto=auto")
    endif()
    add_compile_options(-Wpedantic -Wall)
endif()


if(MSVC)
    add_compile_options(/W4)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wpedantic -Wall)
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

# Windows networking libraries for ASIO
if(WIN32)
    set(WINDOWS_LIBS ws2_32 wsock32 mswsock)
    if(MINGW)
        list(APPEND WINDOWS_LIBS iphlpapi)
    endif()
endif()

# template libraries instantiations
add_subdirectory(src) 

function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    if(ENABLE_CUDA)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    endif()
    
    target_link_libraries(${target_name} PRIVATE dcnn_lib)
    
    if(ENABLE_TBB)
        target_link_libraries(${target_name} PRIVATE TBB::tbb)
    endif()
    
    if(WIN32)
        target_link_libraries(${target_name} PRIVATE ${WINDOWS_LIBS})
    endif()
endfunction()

add_subdirectory(examples)
add_subdirectory(benchmarks)