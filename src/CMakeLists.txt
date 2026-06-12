# SYNET CMakeLists.txt
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Type module
file(GLOB_RECURSE SYNET_TYPE_SOURCES 
    "type/*.cpp"
)
add_library(synet_type STATIC ${SYNET_TYPE_SOURCES})
target_include_directories(synet_type PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    target_link_libraries(synet_type PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Macros module
file(GLOB_RECURSE SYNET_MACROS_SOURCES 
    "macros/*.cpp"
)
add_library(synet_macros STATIC ${SYNET_MACROS_SOURCES})
target_include_directories(synet_macros PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_macros PUBLIC synet_type)

# CUDA module
if(ENABLE_CUDA)
    file(GLOB_RECURSE SYNET_CUDA_SOURCES 
        "cuda/*.cu"
    )
    add_library(synet_cuda STATIC ${SYNET_CUDA_SOURCES})
    target_include_directories(synet_cuda PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # enable separable compilation for CUDA device code linking
    set_target_properties(synet_cuda PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(synet_cuda PUBLIC synet_type)
    target_link_libraries(synet_cuda PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(synet_cuda PUBLIC CUDA::cudnn cudnn_frontend CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Logging module
file(GLOB_RECURSE SYNET_LOGGING_SOURCES 
    "logging/*.cpp"
)
add_library(synet_logging STATIC ${SYNET_LOGGING_SOURCES})
target_include_directories(synet_logging PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_logging PUBLIC spdlog::spdlog)

# Device module
file(GLOB_RECURSE SYNET_DEVICE_SOURCES 
    "device/*.cpp"
)
add_library(synet_device STATIC ${SYNET_DEVICE_SOURCES})
target_include_directories(synet_device PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    <INSTALL_INTERFACE:include>
)
target_link_libraries(synet_device PUBLIC spdlog::spdlog)
if(ENABLE_CUDA)
    target_link_libraries(synet_device PUBLIC CUDA::cudart CUDA::cublas)
endif()
target_link_libraries(synet_device PUBLIC synet_type synet_logging)

# Threading module
file(GLOB_RECURSE SYNET_THREADING_SOURCES 
    "threading/*.cpp"
)
add_library(synet_threading STATIC ${SYNET_THREADING_SOURCES})
target_include_directories(synet_threading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Common module
file(GLOB_RECURSE SYNET_COMMON_SOURCES 
    "common/*.cpp"
)
add_library(synet_common STATIC ${SYNET_COMMON_SOURCES})
target_include_directories(synet_common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_common PUBLIC synet_type)
target_link_libraries(synet_common PUBLIC nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_link_libraries(synet_common PUBLIC m)
endif()

# Operations Module
file(GLOB_RECURSE SYNET_OPS_SOURCES 
    "ops/*.cpp"
    "ops/*.cu"
)
add_library(synet_ops STATIC ${SYNET_OPS_SOURCES})
target_include_directories(synet_ops PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_ops PUBLIC synet_device synet_threading synet_type)
if(ENABLE_CUDA)
    target_link_libraries(synet_ops PUBLIC synet_cuda CUDA::cudart CUDA::cublas)
endif()

# Matrix module
file(GLOB_RECURSE SYNET_MATRIX_SOURCES 
    "matrix/*.cpp"
)
add_library(synet_matrix STATIC ${SYNET_MATRIX_SOURCES})
target_include_directories(synet_matrix PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_matrix PUBLIC synet_device synet_ops synet_type)
if(NOT MSVC)
    target_link_libraries(synet_matrix PUBLIC m)
endif()

# Tensor module
file(GLOB_RECURSE SYNET_TENSOR_SOURCES 
    "tensor/*.cpp"
    "tensor/*.cu"
)
add_library(synet_tensor STATIC ${SYNET_TENSOR_SOURCES})
target_include_directories(synet_tensor PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_tensor PUBLIC synet_matrix synet_ops synet_device synet_type)
if(NOT MSVC)
    target_link_libraries(synet_tensor PUBLIC m)
endif()
if(ENABLE_CUDA)
    target_link_libraries(synet_tensor PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Utilities module
file(GLOB_RECURSE SYNET_UTILS_SOURCES 
    "utils/*.cpp"
    "utils/*.cu"
)
add_library(synet_utils STATIC ${SYNET_UTILS_SOURCES})
target_include_directories(synet_utils PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        file(GLOB ASM_SOURCES "utils/*.asm")
        target_sources(synet_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM_NASM)
        endforeach()
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        file(GLOB ASM_SOURCES "utils/*.S")
        target_sources(synet_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM)
        endforeach()
        message(STATUS "Using GNU assembly optimizations")
    else()
        message(STATUS "Assembly optimizations disabled")
    endif()
endif()

# add platform-specific lib for hardware info
if(NOT MSVC)
    target_link_libraries(synet_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(synet_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(synet_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(synet_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(synet_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(synet_utils PUBLIC pdh)
endif()
if(ENABLE_CUDA)
    target_link_libraries(synet_utils PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Data loading library
file(GLOB_RECURSE SYNET_DATA_LOADING_SOURCES 
    "data_loading/*.cpp"
)
add_library(synet_data_loading STATIC ${SYNET_DATA_LOADING_SOURCES})
target_include_directories(synet_data_loading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_data_loading PUBLIC synet_utils synet_tensor synet_matrix synet_type)
target_link_libraries(synet_data_loading PUBLIC nlohmann_json::nlohmann_json)

# Data augmentation library
file(GLOB_RECURSE SYNET_DATA_AUGMENTATION_SOURCES
    "data_augmentation/*.cpp"
)
add_library(synet_data_augmentation STATIC ${SYNET_DATA_AUGMENTATION_SOURCES})
target_include_directories(synet_data_augmentation PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_data_augmentation PUBLIC synet_tensor synet_type)

# Math library
file(GLOB_RECURSE SYNET_MATH_SOURCES 
    "math/*.cpp"
)
if(ENABLE_CUDA)
    file(GLOB_RECURSE SYNET_MATH_CUDA_SOURCES 
        "math/*.cu"
    )
    add_library(synet_math STATIC ${SYNET_MATH_SOURCES} ${SYNET_MATH_CUDA_SOURCES})
else()
    add_library(synet_math STATIC ${SYNET_MATH_SOURCES})
endif()
target_include_directories(synet_math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_math PUBLIC synet_utils synet_type)
if(ENABLE_CUDA)
    set_target_properties(synet_math PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
    target_link_libraries(synet_math PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(synet_math PUBLIC CUDA::cudnn cudnn_frontend CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Neural Network library
file(GLOB_RECURSE SYNET_NN_SOURCES 
    "nn/*.cpp"
)
file(GLOB_RECURSE SYNET_NN_CUDA_SOURCES 
    "nn/*.cu"
)
add_library(synet_nn STATIC ${SYNET_NN_SOURCES} ${SYNET_NN_CUDA_SOURCES})
target_include_directories(synet_nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_nn PUBLIC 
    synet_tensor synet_utils synet_data_loading synet_data_augmentation synet_math synet_device synet_ops synet_threading synet_type synet_logging synet_common
)
if(ENABLE_CUDA)
    target_link_libraries(synet_nn PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(synet_nn PUBLIC CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Profiling module
file(GLOB_RECURSE SYNET_PROFILING_SOURCES
    "profiling/*.cpp"
)
add_library(synet_profiling STATIC ${SYNET_PROFILING_SOURCES})
target_include_directories(synet_profiling PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_profiling PUBLIC synet_utils synet_logging)

# Partitioner module
file(GLOB_RECURSE SYNET_PARTITIONER_SOURCES 
    "partitioner/*.cpp"
)
add_library(synet_partitioner STATIC ${SYNET_PARTITIONER_SOURCES})
target_include_directories(synet_partitioner PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(synet_partitioner PUBLIC synet_nn)

# Distributed module
file(GLOB_RECURSE SYNET_DISTRIBUTED_SOURCES 
    "distributed/*.cpp"
)
add_library(synet_distributed STATIC ${SYNET_DISTRIBUTED_SOURCES})
target_include_directories(synet_distributed PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)
# Private include for build-time dependencies
target_include_directories(synet_distributed PRIVATE 
    ${asio_SOURCE_DIR}/asio/include
    ${IBVERBS_INCLUDE_DIRS}
)
target_link_libraries(synet_distributed PUBLIC synet_nn synet_partitioner synet_logging synet_profiling synet_type) 
target_link_libraries(synet_distributed PUBLIC nlohmann_json::nlohmann_json ${IBVERBS_LIBRARIES})
target_compile_definitions(synet_distributed PUBLIC ASIO_STANDALONE)

# Main SYNET lib
set(SYNET_MODULES synet_type synet_device synet_threading synet_ops synet_logging synet_matrix synet_tensor synet_data_augmentation synet_data_loading synet_nn synet_utils synet_math synet_distributed synet_profiling synet_partitioner)
if(ENABLE_CUDA)
    list(APPEND SYNET_MODULES synet_cuda)
endif()

# Create the main shared library with entry point source file
add_library(synet_lib SHARED synet_lib.cpp)

# Set library output name (libsynet.so on Linux, synet.dll on Windows)
set_target_properties(synet_lib PROPERTIES
    OUTPUT_NAME "synet"
    VERSION 0.1.0
    SOVERSION 1
)

set_target_properties(synet_lib PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Include directories
target_include_directories(synet_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link all internal static modules into the shared library
target_link_libraries(synet_lib PUBLIC ${SYNET_MODULES})

# Platform-specific system libraries
if(NOT MSVC)
    target_link_libraries(synet_lib PUBLIC m pthread)
endif()

if(WIN32)
    target_link_libraries(synet_lib PUBLIC ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(synet_lib PUBLIC iphlpapi)
    endif()
endif()


# Link external dependencies to static modules
foreach(lib ${SYNET_MODULES})
    link_tbb(PUBLIC ${lib})
    link_cuda(PUBLIC ${lib})
    link_mkl(PUBLIC ${lib})
    link_dnnl(PUBLIC ${lib})
    link_openmp(PUBLIC ${lib})
endforeach()

# Link external dependencies to the main shared library
link_tbb(PUBLIC synet_lib)
link_cuda(PUBLIC synet_lib)
link_mkl(PUBLIC synet_lib)
link_dnnl(PUBLIC synet_lib)
link_openmp(PUBLIC synet_lib)