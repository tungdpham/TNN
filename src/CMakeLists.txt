# tunx CMakeLists.txt
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Type module
file(GLOB_RECURSE TUNX_TYPE_SOURCES 
    "type/*.cpp"
)
add_library(tunx_type STATIC ${TUNX_TYPE_SOURCES})
target_include_directories(tunx_type PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    target_link_libraries(tunx_type PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Macros module
file(GLOB_RECURSE TUNX_MACROS_SOURCES 
    "macros/*.cpp"
)
add_library(tunx_macros STATIC ${TUNX_MACROS_SOURCES})
target_include_directories(tunx_macros PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_macros PUBLIC tunx_type)

# CUDA module
if(ENABLE_CUDA)
    file(GLOB_RECURSE TUNX_CUDA_SOURCES 
        "cuda/*.cu"
    )
    add_library(tunx_cuda STATIC ${TUNX_CUDA_SOURCES})
    target_include_directories(tunx_cuda PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    # enable separable compilation for CUDA device code linking
    set_target_properties(tunx_cuda PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(tunx_cuda PUBLIC tunx_type)
    target_link_libraries(tunx_cuda PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(tunx_cuda PUBLIC CUDA::cudnn cudnn_frontend CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Logging module
file(GLOB_RECURSE TUNX_LOGGING_SOURCES 
    "logging/*.cpp"
)
add_library(tunx_logging STATIC ${TUNX_LOGGING_SOURCES})
target_include_directories(tunx_logging PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_logging PUBLIC spdlog::spdlog)

# Device module
file(GLOB_RECURSE TUNX_DEVICE_SOURCES 
    "device/*.cpp"
)
add_library(tunx_device STATIC ${TUNX_DEVICE_SOURCES})
target_include_directories(tunx_device PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_device PUBLIC spdlog::spdlog)
if(ENABLE_CUDA)
    target_link_libraries(tunx_device PUBLIC CUDA::cudart CUDA::cublas)
endif()
target_link_libraries(tunx_device PUBLIC tunx_type tunx_logging)

# Threading module
file(GLOB_RECURSE TUNX_THREADING_SOURCES 
    "threading/*.cpp"
)
add_library(tunx_threading STATIC ${TUNX_THREADING_SOURCES})
target_include_directories(tunx_threading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Common module
file(GLOB_RECURSE TUNX_COMMON_SOURCES 
    "common/*.cpp"
)
add_library(tunx_common STATIC ${TUNX_COMMON_SOURCES})
target_include_directories(tunx_common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_common PUBLIC tunx_type)
target_link_libraries(tunx_common PUBLIC nlohmann_json::nlohmann_json)
if(NOT MSVC)
    target_link_libraries(tunx_common PUBLIC m)
endif()

# Operations Module
file(GLOB_RECURSE TUNX_OPS_SOURCES 
    "ops/*.cpp"
    "ops/*.cu"
)
add_library(tunx_ops STATIC ${TUNX_OPS_SOURCES})
target_include_directories(tunx_ops PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    set_target_properties(tunx_ops PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
endif()
target_link_libraries(tunx_ops PUBLIC tunx_device tunx_threading tunx_type)
if(ENABLE_CUDA)
    target_link_libraries(tunx_ops PUBLIC tunx_cuda CUDA::cudart CUDA::cublas)
endif()

# Matrix module
file(GLOB_RECURSE TUNX_MATRIX_SOURCES 
    "matrix/*.cpp"
)
add_library(tunx_matrix STATIC ${TUNX_MATRIX_SOURCES})
target_include_directories(tunx_matrix PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_matrix PUBLIC tunx_device tunx_ops tunx_type)
if(NOT MSVC)
    target_link_libraries(tunx_matrix PUBLIC m)
endif()

# Tensor module
file(GLOB_RECURSE TUNX_TENSOR_SOURCES 
    "tensor/*.cpp"
    "tensor/*.cu"
)
add_library(tunx_tensor STATIC ${TUNX_TENSOR_SOURCES})
target_include_directories(tunx_tensor PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    set_target_properties(tunx_tensor PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
endif()
target_link_libraries(tunx_tensor PUBLIC tunx_matrix tunx_ops tunx_device tunx_type)
if(NOT MSVC)
    target_link_libraries(tunx_tensor PUBLIC m)
endif()
if(ENABLE_CUDA)
    target_link_libraries(tunx_tensor PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Utilities module
file(GLOB_RECURSE TUNX_UTILS_SOURCES 
    "utils/*.cpp"
    "utils/*.cu"
)
add_library(tunx_utils STATIC ${TUNX_UTILS_SOURCES})
target_include_directories(tunx_utils PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    set_target_properties(tunx_utils PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
endif()

# add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        file(GLOB ASM_SOURCES "utils/*.asm")
        target_sources(tunx_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM_NASM)
        endforeach()
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        file(GLOB ASM_SOURCES "utils/*.S")
        target_sources(tunx_utils PRIVATE ${ASM_SOURCES})
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
    target_link_libraries(tunx_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(tunx_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(tunx_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(tunx_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(tunx_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(tunx_utils PUBLIC pdh)
endif()
if(ENABLE_CUDA)
    target_link_libraries(tunx_utils PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Data loading library
file(GLOB_RECURSE TUNX_DATA_LOADING_SOURCES 
    "data_loading/*.cpp"
)
add_library(tunx_data_loading STATIC ${TUNX_DATA_LOADING_SOURCES})
target_include_directories(tunx_data_loading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_data_loading PUBLIC tunx_utils tunx_tensor tunx_matrix tunx_type)
target_link_libraries(tunx_data_loading PUBLIC nlohmann_json::nlohmann_json)

# Data augmentation library
file(GLOB_RECURSE TUNX_DATA_AUGMENTATION_SOURCES
    "data_augmentation/*.cpp"
)
add_library(tunx_data_augmentation STATIC ${TUNX_DATA_AUGMENTATION_SOURCES})
target_include_directories(tunx_data_augmentation PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_data_augmentation PUBLIC tunx_tensor tunx_type)

# Math library
file(GLOB_RECURSE TUNX_MATH_SOURCES 
    "math/*.cpp"
)
if(ENABLE_CUDA)
    file(GLOB_RECURSE TUNX_MATH_CUDA_SOURCES 
        "math/*.cu"
    )
    add_library(tunx_math STATIC ${TUNX_MATH_SOURCES} ${TUNX_MATH_CUDA_SOURCES})
else()
    add_library(tunx_math STATIC ${TUNX_MATH_SOURCES})
endif()
target_include_directories(tunx_math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_math PUBLIC tunx_utils tunx_type)
if(ENABLE_CUDA)
    set_target_properties(tunx_math PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
    target_link_libraries(tunx_math PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(tunx_math PUBLIC CUDA::cudnn cudnn_frontend CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Neural Network library
file(GLOB_RECURSE TUNX_NN_SOURCES 
    "nn/*.cpp"
)
file(GLOB_RECURSE TUNX_NN_CUDA_SOURCES 
    "nn/*.cu"
)
add_library(tunx_nn STATIC ${TUNX_NN_SOURCES} ${TUNX_NN_CUDA_SOURCES})
target_include_directories(tunx_nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    set_target_properties(tunx_nn PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 20
    )
endif()
target_link_libraries(tunx_nn PUBLIC 
    tunx_tensor tunx_utils tunx_data_loading tunx_data_augmentation tunx_math tunx_device tunx_ops tunx_threading tunx_type tunx_logging tunx_common
)
if(ENABLE_CUDA)
    target_link_libraries(tunx_nn PUBLIC CUDA::cudart CUDA::cublas)
    if(ENABLE_CUDNN)
        target_link_libraries(tunx_nn PUBLIC CUDA::nvrtc CUDA::cuda_driver)
    endif()
endif()

# Profiling module
file(GLOB_RECURSE TUNX_PROFILING_SOURCES
    "profiling/*.cpp"
)
add_library(tunx_profiling STATIC ${TUNX_PROFILING_SOURCES})
target_include_directories(tunx_profiling PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_profiling PUBLIC tunx_utils tunx_logging)

# Partitioner module
file(GLOB_RECURSE TUNX_PARTITIONER_SOURCES 
    "partitioner/*.cpp"
)
add_library(tunx_partitioner STATIC ${TUNX_PARTITIONER_SOURCES})
target_include_directories(tunx_partitioner PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tunx_partitioner PUBLIC tunx_nn)

# Distributed module
file(GLOB_RECURSE TUNX_DISTRIBUTED_SOURCES 
    "distributed/*.cpp"
)
add_library(tunx_distributed STATIC ${TUNX_DISTRIBUTED_SOURCES})
target_include_directories(tunx_distributed PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(tunx_distributed PRIVATE 
    ${asio_SOURCE_DIR}/asio/include
    ${IBVERBS_INCLUDE_DIRS}
)
target_link_libraries(tunx_distributed PUBLIC tunx_nn tunx_partitioner tunx_logging tunx_profiling tunx_type) 
target_link_libraries(tunx_distributed PUBLIC nlohmann_json::nlohmann_json ${IBVERBS_LIBRARIES})
target_compile_definitions(tunx_distributed PUBLIC ASIO_STANDALONE)

# Main consolidated module setup
set(tunx_MODULES tunx_type tunx_device tunx_threading tunx_ops tunx_logging tunx_matrix tunx_tensor tunx_data_augmentation tunx_data_loading tunx_nn tunx_utils tunx_math tunx_distributed tunx_profiling tunx_partitioner)
if(ENABLE_CUDA)
    list(APPEND tunx_MODULES tunx_cuda)
endif()

# Create main wrapper library
add_library(tunx_lib SHARED tunx_lib.cpp)

set_target_properties(tunx_lib PROPERTIES
    OUTPUT_NAME "tunx"
    VERSION 0.1.0
    SOVERSION 1
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(tunx_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tunx_lib PUBLIC ${tunx_MODULES})

if(NOT MSVC)
    target_link_libraries(tunx_lib PUBLIC m pthread)
endif()

if(WIN32)
    target_link_libraries(tunx_lib PUBLIC ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(tunx_lib PUBLIC iphlpapi)
    endif()
endif()

foreach(lib ${tunx_MODULES})
    link_tbb(PRIVATE ${lib})
    link_cuda(PRIVATE ${lib})
    link_mkl(PRIVATE ${lib})
    link_dnnl(PRIVATE ${lib})
    link_openmp(PRIVATE ${lib})
endforeach()

link_cuda(PUBLIC tunx_lib)
link_mkl(PUBLIC tunx_lib)
link_dnnl(PUBLIC tunx_lib)
link_openmp(PUBLIC tunx_lib)
link_tbb(PUBLIC tunx_lib)