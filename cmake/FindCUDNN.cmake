# Auto-detect and enable cuDNN if available
if(NOT DEFINED ENABLE_CUDNN OR ENABLE_CUDNN)
    # Try to find cuDNN library
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR}
        PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
        /usr/include/x86_64-linux-gnu
    )
    
    if(CUDNN_LIBRARY)
        message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")
        set(ENABLE_CUDNN ON CACHE BOOL "Enable cuDNN support (requires CUDA)" FORCE)
        add_library(CUDA::cudnn UNKNOWN IMPORTED)
        set_target_properties(CUDA::cudnn PROPERTIES
            IMPORTED_LOCATION ${CUDNN_LIBRARY}
        )
        
        if(CUDNN_INCLUDE_DIR)
            file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_H_CONTENTS)
            string(REGEX MATCH "#define CUDNN_MAJOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
            set(CUDNN_MAJOR_VERSION ${CMAKE_MATCH_1})
            string(REGEX MATCH "#define CUDNN_MINOR ([0-9]+)" _ "${CUDNN_H_CONTENTS}")
            set(CUDNN_MINOR_VERSION ${CMAKE_MATCH_1})
        endif()

        message(STATUS "cuDNN MAJOR version: ${CUDNN_MAJOR_VERSION}")
        message(STATUS "cuDNN MINOR version: ${CUDNN_MINOR_VERSION}")
    else()
        message(STATUS "cuDNN library not found.")
        set(ENABLE_CUDNN OFF CACHE BOOL "Enable cuDNN support (requires CUDA)" FORCE)
    endif()
endif()
