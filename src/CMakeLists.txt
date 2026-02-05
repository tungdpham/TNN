# TNN CMakeLists.txt
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Type module
file(GLOB_RECURSE TNN_TYPE_SOURCES 
    "type/*.cpp"
)
add_library(tnn_type STATIC ${TNN_TYPE_SOURCES})
target_include_directories(tnn_type PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    target_link_libraries(tnn_type PUBLIC CUDA::cudart CUDA::cublas)
endif()

# CUDA module
if(ENABLE_CUDA)
    file(GLOB_RECURSE TNN_CUDA_SOURCES 
        "cuda/*.cu"
    )
    add_library(tnn_cuda STATIC ${TNN_CUDA_SOURCES})
    target_include_directories(tnn_cuda PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # enable separable compilation for CUDA device code linking
    set_target_properties(tnn_cuda PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(tnn_cuda PUBLIC tnn_type)
    target_link_libraries(tnn_cuda PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(tnn_cuda PUBLIC CUDA::cudnn cudnn_frontend)
    endif()
endif()

# Device module
file(GLOB_RECURSE TNN_DEVICE_SOURCES 
    "device/*.cpp"
)
add_library(tnn_device STATIC ${TNN_DEVICE_SOURCES})
target_include_directories(tnn_device PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    target_link_libraries(tnn_device PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Threading module
file(GLOB_RECURSE TNN_THREADING_SOURCES 
    "threading/*.cpp"
)
add_library(tnn_threading STATIC ${TNN_THREADING_SOURCES})
target_include_directories(tnn_threading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Common module
file(GLOB_RECURSE TNN_COMMON_SOURCES 
    "common/*.cpp"
)
add_library(tnn_common STATIC ${TNN_COMMON_SOURCES})
target_include_directories(tnn_common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_common PUBLIC tnn_type)
target_link_libraries(tnn_common PUBLIC nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_link_libraries(tnn_common PUBLIC m)
endif()

# Operations Module
file(GLOB_RECURSE TNN_OPS_SOURCES 
    "ops/*.cpp"
    "ops/*.cu"
)
add_library(tnn_ops STATIC ${TNN_OPS_SOURCES})
target_include_directories(tnn_ops PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_ops PUBLIC tnn_device tnn_threading tnn_type)
if(ENABLE_CUDA)
    target_link_libraries(tnn_ops PUBLIC tnn_cuda CUDA::cudart CUDA::cublas)
endif()

# Matrix module
file(GLOB_RECURSE TNN_MATRIX_SOURCES 
    "matrix/*.cpp"
)
add_library(tnn_matrix STATIC ${TNN_MATRIX_SOURCES})
target_include_directories(tnn_matrix PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_matrix PUBLIC tnn_device tnn_ops tnn_type)
if(NOT MSVC)
    target_link_libraries(tnn_matrix PUBLIC m)
endif()

# Tensor module
file(GLOB_RECURSE TNN_TENSOR_SOURCES 
    "tensor/*.cpp"
    "tensor/*.cu"
)
add_library(tnn_tensor STATIC ${TNN_TENSOR_SOURCES})
target_include_directories(tnn_tensor PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_tensor PUBLIC tnn_matrix tnn_ops tnn_device tnn_type)
if(NOT MSVC)
    target_link_libraries(tnn_tensor PUBLIC m)
endif()
if(ENABLE_CUDA)
    target_link_libraries(tnn_tensor PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Logging module
file(GLOB_RECURSE TNN_LOGGING_SOURCES 
    "logging/*.cpp"
)
add_library(tnn_logging STATIC ${TNN_LOGGING_SOURCES})
target_include_directories(tnn_logging PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_logging PUBLIC spdlog::spdlog)

# Utilities module
file(GLOB_RECURSE TNN_UTILS_SOURCES 
    "utils/*.cpp"
    "utils/*.cu"
)
add_library(tnn_utils STATIC ${TNN_UTILS_SOURCES})
target_include_directories(tnn_utils PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        file(GLOB ASM_SOURCES "utils/*.asm")
        target_sources(tnn_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM_NASM)
        endforeach()
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        file(GLOB ASM_SOURCES "utils/*.S")
        target_sources(tnn_utils PRIVATE ${ASM_SOURCES})
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
    target_link_libraries(tnn_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(tnn_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(tnn_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(tnn_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(tnn_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(tnn_utils PUBLIC pdh)
endif()
if(ENABLE_CUDA)
    target_link_libraries(tnn_utils PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Data loading library
file(GLOB_RECURSE TNN_DATA_LOADING_SOURCES 
    "data_loading/*.cpp"
)
add_library(tnn_data_loading STATIC ${TNN_DATA_LOADING_SOURCES})
target_include_directories(tnn_data_loading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_data_loading PUBLIC tnn_utils tnn_tensor tnn_matrix tnn_type)

# Data augmentation library
file(GLOB_RECURSE TNN_DATA_AUGMENTATION_SOURCES
    "data_augmentation/*.cpp"
)
add_library(tnn_data_augmentation STATIC ${TNN_DATA_AUGMENTATION_SOURCES})
target_include_directories(tnn_data_augmentation PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_data_augmentation PUBLIC tnn_tensor tnn_type)

# Math library
file(GLOB_RECURSE TNN_MATH_SOURCES 
    "math/*.cpp"
)
if(ENABLE_CUDA)
    file(GLOB_RECURSE TNN_MATH_CUDA_SOURCES 
        "math/*.cu"
    )
    add_library(tnn_math STATIC ${TNN_MATH_SOURCES} ${TNN_MATH_CUDA_SOURCES})
else()
    add_library(tnn_math STATIC ${TNN_MATH_SOURCES})
endif()
target_include_directories(tnn_math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_math PUBLIC tnn_utils tnn_type)
if(ENABLE_CUDA)
    set_target_properties(tnn_math PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
    target_link_libraries(tnn_math PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(tnn_math PUBLIC CUDA::cudnn cudnn_frontend)
    endif()
endif()

# Neural Network library
file(GLOB_RECURSE TNN_NN_SOURCES 
    "nn/*.cpp"
)
file(GLOB_RECURSE TNN_NN_CUDA_SOURCES 
    "nn/*.cu"
)
add_library(tnn_nn STATIC ${TNN_NN_SOURCES} ${TNN_NN_CUDA_SOURCES})
target_include_directories(tnn_nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_nn PUBLIC 
    tnn_tensor tnn_utils tnn_data_loading tnn_data_augmentation tnn_math tnn_device tnn_ops tnn_threading tnn_type tnn_logging tnn_common
)
if(ENABLE_CUDA)
    target_link_libraries(tnn_nn PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Profiling module
file(GLOB_RECURSE TNN_PROFILING_SOURCES
    "profiling/*.cpp"
)
add_library(tnn_profiling STATIC ${TNN_PROFILING_SOURCES})
target_include_directories(tnn_profiling PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_profiling PUBLIC tnn_utils tnn_logging)

# Partitioner module
file(GLOB_RECURSE TNN_PARTITIONER_SOURCES 
    "partitioner/*.cpp"
)
add_library(tnn_partitioner STATIC ${TNN_PARTITIONER_SOURCES})
target_include_directories(tnn_partitioner PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_partitioner PUBLIC tnn_nn)

# Distributed module
file(GLOB_RECURSE TNN_DISTRIBUTED_SOURCES 
    "distributed/*.cpp"
)
add_library(tnn_distributed STATIC ${TNN_DISTRIBUTED_SOURCES})
target_include_directories(tnn_distributed PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)
# Private include for build-time dependencies
target_include_directories(tnn_distributed PRIVATE 
    ${asio_SOURCE_DIR}/asio/include
    ${IBVERBS_INCLUDE_DIRS}
)
target_link_libraries(tnn_distributed PUBLIC tnn_nn tnn_partitioner tnn_logging tnn_profiling tnn_type) 
target_link_libraries(tnn_distributed PUBLIC nlohmann_json::nlohmann_json ${IBVERBS_LIBRARIES})
target_compile_definitions(tnn_distributed PUBLIC ASIO_STANDALONE)

# Main TNN library - Single SHARED library bundling all static modules
set(TNN_MODULES tnn_type tnn_device tnn_threading tnn_ops tnn_logging tnn_matrix tnn_tensor tnn_data_augmentation tnn_data_loading tnn_nn tnn_utils tnn_math tnn_distributed tnn_profiling tnn_partitioner)
if(ENABLE_CUDA)
    list(APPEND TNN_MODULES tnn_cuda)
endif()

# Create the main shared library with entry point source file
add_library(tnn_lib SHARED tnn_lib.cpp)

# Set library output name (libtnn.so on Linux, tnn.dll on Windows)
set_target_properties(tnn_lib PROPERTIES
    OUTPUT_NAME "tnn"
    VERSION 0.1.0
    SOVERSION 1
)

# Configure symbol visibility for cleaner API
set_target_properties(tnn_lib PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Include directories
target_include_directories(tnn_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link all internal static modules into the shared library
target_link_libraries(tnn_lib PUBLIC ${TNN_MODULES})

# Platform-specific system libraries
if(NOT MSVC)
    target_link_libraries(tnn_lib PUBLIC m pthread)
endif()

if(WIN32)
    target_link_libraries(tnn_lib PUBLIC ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(tnn_lib PUBLIC iphlpapi)
    endif()
endif()


# Link external dependencies to static modules
foreach(lib ${TNN_MODULES})
    link_tbb(PUBLIC ${lib})
    link_cuda(PUBLIC ${lib})
    link_mkl(PUBLIC ${lib})
    link_openmp(PUBLIC ${lib})
endforeach()

# Link external dependencies to the main shared library
link_tbb(PUBLIC tnn_lib)
link_cuda(PUBLIC tnn_lib)
link_mkl(PUBLIC tnn_lib)
link_openmp(PUBLIC tnn_lib)