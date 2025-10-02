
add_library(dcnn_matrix STATIC matrix.cpp)
target_include_directories(dcnn_matrix PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(NOT MSVC)
    target_link_libraries(dcnn_matrix PUBLIC m)
endif()

add_library(dcnn_tensor STATIC tensor.cpp)
target_include_directories(dcnn_tensor PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(NOT MSVC)
    target_link_libraries(dcnn_tensor PUBLIC dcnn_matrix m)
else()
    target_link_libraries(dcnn_tensor PUBLIC dcnn_matrix)
endif()

add_library(dcnn_utils STATIC utils.cpp hardware_info.cpp)
target_include_directories(dcnn_utils PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        file(GLOB ASM_SOURCES "utils/*.asm")
        target_sources(dcnn_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM_NASM)
        endforeach()
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        file(GLOB ASM_SOURCES "utils/*.S")
        target_sources(dcnn_utils PRIVATE ${ASM_SOURCES})
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
    target_link_libraries(dcnn_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(dcnn_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(dcnn_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(dcnn_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(dcnn_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(dcnn_utils PUBLIC pdh)
endif()

add_library(dcnn_data_loading STATIC data_loading.cpp)
target_include_directories(dcnn_data_loading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dcnn_data_loading PUBLIC dcnn_utils dcnn_tensor dcnn_matrix)

file(GLOB_RECURSE DCNN_MATH_SOURCES 
    "math/*.cpp"
)
add_library(dcnn_math STATIC ${DCNN_MATH_SOURCES})
target_include_directories(dcnn_math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(dcnn_math PUBLIC dcnn_utils)

file(GLOB_RECURSE DCNN_NN_SOURCES 
    "nn.cpp"
    "nn/*.cpp"
)

add_library(dcnn_nn STATIC ${DCNN_NN_SOURCES})
target_include_directories(dcnn_nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(NOT MSVC)
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils dcnn_data_loading m)
else()
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils dcnn_data_loading)
endif()
target_link_libraries(dcnn_nn PUBLIC nlohmann_json::nlohmann_json)

add_library(dcnn_pipeline STATIC pipeline.cpp)
target_include_directories(dcnn_pipeline PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)
# Private include for build-time dependencies
target_include_directories(dcnn_pipeline PRIVATE 
    ${asio_SOURCE_DIR}/asio/include
)
target_link_libraries(dcnn_pipeline PUBLIC dcnn_nn protobuf::libprotobuf)
target_link_libraries(dcnn_pipeline PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(dcnn_pipeline PUBLIC libzstd_static)
target_compile_definitions(dcnn_pipeline PUBLIC ASIO_STANDALONE)

set(PROTO_DIR "${CMAKE_SOURCE_DIR}/include/pipeline/proto")
# auto generate proto files
if(EXISTS "${PROTO_DIR}")
    file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")
    if(PROTO_FILES)
        set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
        file(MAKE_DIRECTORY ${PROTO_OUT_DIR})
        
        protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
        
        set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES
            GENERATED TRUE
        )
        
        add_library(dcnn_proto STATIC ${PROTO_SRCS})
        target_link_libraries(dcnn_proto PUBLIC protobuf::libprotobuf)
        target_include_directories(dcnn_proto PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )
        
        target_link_libraries(dcnn_pipeline PUBLIC dcnn_proto)
    endif()
endif()

add_library(dcnn_instantiations INTERFACE)
target_link_libraries(dcnn_instantiations INTERFACE dcnn_pipeline dcnn_nn dcnn_data_loading dcnn_math dcnn_tensor dcnn_matrix dcnn_utils)

add_library(dcnn_lib INTERFACE)
target_include_directories(dcnn_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(NOT MSVC)
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations m pthread)
else()
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations)
endif()

target_compile_definitions(dcnn_lib INTERFACE ASIO_STANDALONE)

if(WIN32)
    target_link_libraries(dcnn_lib INTERFACE ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(dcnn_lib INTERFACE iphlpapi)
    endif()
endif()

set(INSTANTIATION_LIBS dcnn_matrix dcnn_tensor dcnn_nn dcnn_utils dcnn_math)

if(ENABLE_OPENMP)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC OpenMP::OpenMP_CXX)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC TBB::tbb TBB::tbbmalloc)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE TBB::tbb TBB::tbbmalloc)
endif()

if(ENABLE_CUDA)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE CUDA::cudart CUDA::cublas CUDA::curand)
endif()

if(ENABLE_MKL)
    foreach(lib ${INSTANTIATION_LIBS})
        if(TARGET MKL::MKL)
            target_link_libraries(${lib} PUBLIC MKL::MKL)
        else()
            target_include_directories(${lib} PUBLIC ${MKL_INCLUDE_DIRS})
            target_link_libraries(${lib} PUBLIC ${MKL_LIBRARIES})
        endif()
    endforeach()
    
    if(TARGET MKL::MKL)
        target_link_libraries(dcnn_lib INTERFACE MKL::MKL)
    else()
        target_link_libraries(dcnn_lib INTERFACE ${MKL_LIBRARIES})
    endif()
endif()