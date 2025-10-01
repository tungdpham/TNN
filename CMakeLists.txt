cmake_minimum_required(VERSION 3.18)

project(DCNN LANGUAGES CXX)

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

# defaults (might change)
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

# FetchContent to manage third-party dependencies
include(FetchContent)

# ASIO (header-only library)
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG asio-1-30-2
)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# zstandard compression library
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.7
    SOURCE_SUBDIR build/cmake
)

# make the dependencies available
FetchContent_MakeAvailable(asio nlohmann_json zstd)

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

if(ENABLE_MKL)
    message("Finding Intel MKL")
    
    find_path(MKL_INCLUDE_DIR
        NAMES mkl.h
        PATHS
            /opt/intel/oneapi/mkl/latest/include
            /opt/intel/mkl/include
            $ENV{MKLROOT}/include
    )
    
    find_library(MKL_RT_LIBRARY
        NAMES mkl_rt
        PATHS
            /opt/intel/oneapi/mkl/latest/lib
            /opt/intel/oneapi/mkl/latest/lib/intel64
            /opt/intel/mkl/lib/intel64
            $ENV{MKLROOT}/lib/intel64
            $ENV{MKLROOT}/lib
    )
    
    if(MKL_RT_LIBRARY AND MKL_INCLUDE_DIR)
            set(MKL_FOUND TRUE)
            set(MKL_LIBRARIES ${MKL_RT_LIBRARY})
            set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
            message(STATUS "Found Intel MKL with mkl_rt runtime library")
            message(STATUS "MKL Include: ${MKL_INCLUDE_DIRS}")
            message(STATUS "MKL Libraries: ${MKL_LIBRARIES}")
    else()
        find_library(MKL_CORE_LIBRARY
            NAMES mkl_core
            PATHS
                /opt/intel/oneapi/mkl/latest/lib
                /opt/intel/oneapi/mkl/latest/lib/intel64
                /opt/intel/mkl/lib/intel64
                $ENV{MKLROOT}/lib/intel64
                $ENV{MKLROOT}/lib
        )
        
        find_library(MKL_INTEL_LP64_LIBRARY
            NAMES mkl_intel_lp64
            PATHS
                /opt/intel/oneapi/mkl/latest/lib
                /opt/intel/oneapi/mkl/latest/lib/intel64
                /opt/intel/mkl/lib/intel64
                $ENV{MKLROOT}/lib/intel64
                $ENV{MKLROOT}/lib
        )

        if(ENABLE_TBB)
            message(STATUS "Attempting to link MKL with TBB threading")
            find_library(MKL_THREAD_LIBRARY
                NAMES mkl_tbb_thread
                PATHS
                    /opt/intel/oneapi/mkl/latest/lib/intel64
                    /opt/intel/mkl/lib/intel64
                    $ENV{MKLROOT}/lib/intel64
                    /usr/lib/x86_64-linux-gnu
            )
            
            if(MKL_THREAD_LIBRARY AND TBB_FOUND)
                set(MKL_LINK_THREADING_LIBS ${MKL_THREAD_LIBRARY} TBB::tbb)
            else()
                message(WARNING "MKL TBB threading library not found or TBB not enabled. Falling back to Intel OpenMP.")
                set(ENABLE_TBB OFF)
            endif()
        endif()

        if(NOT MKL_LINK_THREADING_LIBS)
            message(STATUS "Linking MKL with Intel OpenMP threading")
            find_library(MKL_THREAD_LIBRARY
                NAMES mkl_intel_thread
                PATHS
                    /opt/intel/oneapi/mkl/latest/lib/intel64
                    /opt/intel/mkl/lib/intel64
                    $ENV{MKLROOT}/lib/intel64
                    /usr/lib/x86_64-linux-gnu
            )
            if(MKL_THREAD_LIBRARY)
                set(MKL_LINK_THREADING_LIBS ${MKL_THREAD_LIBRARY} iomp5)
            endif()
        endif()
        
        if(MKL_INCLUDE_DIR AND MKL_CORE_LIBRARY AND MKL_INTEL_LP64_LIBRARY AND MKL_LINK_THREADING_LIBS)
            set(MKL_FOUND TRUE)
            if(NOT MSVC)
                set(MKL_LIBRARIES -Wl,--no-as-needed ${MKL_INTEL_LP64_LIBRARY} ${MKL_LINK_THREADING_LIBS} ${MKL_CORE_LIBRARY} -Wl,--as-needed)
            else()
                set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_LINK_THREADING_LIBS})
            endif()
            set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
            message(STATUS "Found Intel MKL manually")
            message(STATUS "MKL Include: ${MKL_INCLUDE_DIRS}")
            message(STATUS "MKL Libraries: ${MKL_LIBRARIES}")
        endif()
    endif()
    
    if(MKL_FOUND)
        add_compile_definitions(USE_MKL)
        message(STATUS "Intel MKL enabled")
    else()
        message(FATAL_ERROR "Intel MKL requested but not found. Please install Intel MKL or set MKLROOT environment variable.")
    endif()
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
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    add_compile_definitions(__AVX2__ __SSE2__)
    if(OpenMP_CXX_FOUND)
        add_compile_options(/openmp:llvm)
    endif()
    # Generate assembly files in debug mode
    if(ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/FA /FAcs)
        message(STATUS "Assembly output enabled (MSVC)")
    endif()
elseif(MINGW)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native")
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    add_compile_options(-Wpedantic -Wall)
    # Generate assembly files in debug mode
    if(ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-save-temps=obj -masm=intel -fno-lto)
        message(STATUS "Assembly output enabled (MinGW)")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -march=native -flto=auto -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g -march=native")
    add_compile_options(-Wpedantic -Wall)
    # Generate assembly files in debug mode
    if(ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-save-temps=obj -masm=intel -fno-lto)
        message(STATUS "Assembly output enabled (GCC/Clang)")
    endif()
endif()

# output dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# libraries
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${asio_SOURCE_DIR}/asio/include
)

# Add global compile definitions for third-party libraries
add_compile_definitions(ASIO_STANDALONE)

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
    
    target_link_libraries(${target_name} PRIVATE dcnn_lib dcnn_pipeline dcnn_nn dcnn_data_loading dcnn_math dcnn_tensor dcnn_matrix dcnn_utils)
    target_link_libraries(${target_name} PRIVATE protobuf::libprotobuf)
    
    target_link_libraries(${target_name} PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(${target_name} PRIVATE libzstd_static)
    
    target_include_directories(${target_name} PRIVATE ${asio_SOURCE_DIR}/asio/include)
    target_compile_definitions(${target_name} PRIVATE ASIO_STANDALONE)
    
    if(ENABLE_TBB)
        target_link_libraries(${target_name} PRIVATE TBB::tbb)
    endif()
    
    if(ENABLE_MKL)
        if(TARGET MKL::MKL)
            target_link_libraries(${target_name} PRIVATE MKL::MKL)
        else()
            target_include_directories(${target_name} PRIVATE ${MKL_INCLUDE_DIRS})
            target_link_libraries(${target_name} PRIVATE ${MKL_LIBRARIES})
            
            if(NOT MSVC)
                target_link_libraries(${target_name} PRIVATE pthread m dl)
            endif()

        endif()
    endif()

    
    if(WIN32)
        target_link_libraries(${target_name} PRIVATE ${WINDOWS_LIBS})
    endif()
endfunction()

add_subdirectory(examples)
add_subdirectory(benchmarks)