find_package(CUDAToolkit REQUIRED)

# Auto-detect CUDA architecture
function(detect_gpu_arch)
    # Try to use nvidia-smi to detect compute capability
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE GPU_COMPUTE_CAP
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(GPU_COMPUTE_CAP)
            # Get the first CUDA's compute capability
            string(REPLACE "\n" ";" GPU_LIST ${GPU_COMPUTE_CAP})
            list(GET GPU_LIST 0 FIRST_GPU_CAP)
            
            # Convert compute capability to architecture number (remove decimal point)
            string(REPLACE "." "" CUDA_ARCH_NUMBER ${FIRST_GPU_CAP})
            
            message(STATUS "Detected CUDA compute capability: ${FIRST_GPU_CAP}")
            message(STATUS "Using CUDA architecture: sm_${CUDA_ARCH_NUMBER}")
            
            set(DETECTED_CUDA_ARCH "sm_${CUDA_ARCH_NUMBER}" PARENT_SCOPE)
            set(DETECTED_CUDA_ARCH_NUMBER "${CUDA_ARCH_NUMBER}" PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Fallback: Try using a simple CUDA program to detect at runtime
    if(CMAKE_CUDA_COMPILER)
        set(CUDA_DETECT_FILE "${CMAKE_BINARY_DIR}/detect_cuda_arch.cu")
        file(WRITE ${CUDA_DETECT_FILE}
            "#include <cuda_runtime.h>
            #include <iostream>
            int main() {
                int deviceCount;
                cudaGetDeviceCount(&deviceCount);
                if (deviceCount > 0) {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, 0);
                    std::cout << prop.major << prop.minor << std::endl;
                }
                return 0;
            }")
        try_run(CUDA_DETECT_RUN_RESULT CUDA_DETECT_COMPILE_RESULT
            ${CMAKE_BINARY_DIR} ${CUDA_DETECT_FILE}
            CMAKE_FLAGS "-DCMAKE_CUDA_STANDARD=20"
            RUN_OUTPUT_VARIABLE CUDA_ARCH_OUTPUT
        )
        
        if(CUDA_DETECT_COMPILE_RESULT AND CUDA_DETECT_RUN_RESULT EQUAL 0)
            string(STRIP ${CUDA_ARCH_OUTPUT} CUDA_ARCH_NUMBER)
            message(STATUS "Detected CUDA architecture via compilation: sm_${CUDA_ARCH_NUMBER}")
            set(DETECTED_CUDA_ARCH "sm_${CUDA_ARCH_NUMBER}" PARENT_SCOPE)
            set(DETECTED_CUDA_ARCH_NUMBER "${CUDA_ARCH_NUMBER}" PARENT_SCOPE)
            return()
        endif()
    endif()
endfunction()

# Allow manual override via command line: -DCUDA_ARCH=sm_86
set(CUDA_ARCH_MAX "89" CACHE STRING "Maximum CUDA architecture number to target")

if(NOT DEFINED CUDA_ARCH)
    detect_gpu_arch()
    set(CUDA_ARCH ${DETECTED_CUDA_ARCH})
    set(CUDA_ARCH_NUMBER ${DETECTED_CUDA_ARCH_NUMBER})
    
    # Apply maximum architecture limit if specified
    if(DEFINED CUDA_ARCH_MAX)
        if(CUDA_ARCH_NUMBER GREATER ${CUDA_ARCH_MAX})
            message(STATUS "CUDA architecture ${CUDA_ARCH_NUMBER} exceeds maximum allowed (${CUDA_ARCH_MAX}). Limiting to sm_${CUDA_ARCH_MAX}")
            set(CUDA_ARCH_NUMBER ${CUDA_ARCH_MAX})
            set(CUDA_ARCH "sm_${CUDA_ARCH_MAX}")
        endif()
    endif()
else()
    message(STATUS "Using manually specified CUDA architecture: ${CUDA_ARCH}")
    string(REPLACE "sm_" "" CUDA_ARCH_NUMBER ${CUDA_ARCH})
endif()

if(CUDA_ARCH_NUMBER)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_NUMBER} CACHE STRING "CUDA target architectures" FORCE)
    set(CUDA_ARCH_NUMBER ${CUDA_ARCH_NUMBER} CACHE STRING "CUDA architecture number" FORCE)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    message(STATUS "Set CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
    message(STATUS "CUDA release flags: ${CMAKE_CUDA_FLAGS_RELEASE}")
    message(STATUS "CUDA debug flags: ${CMAKE_CUDA_FLAGS_DEBUG}")
endif()