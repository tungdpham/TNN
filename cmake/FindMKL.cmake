# FindMKL.cmake - Intel MKL detection module
#
# This module finds Intel Math Kernel Library (MKL) and sets up linking.

message(STATUS "Finding Intel MKL")

# Find MKL include directory
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    PATHS
        /opt/intel/oneapi/mkl/latest/include
        /opt/intel/mkl/include
        $ENV{MKLROOT}/include
)

# Try the simplified runtime library first (mkl_rt)
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
    # Fall back to detailed linking model
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

    # Determine threading library based on TBB availability
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

    # Fall back to Intel OpenMP threading if TBB not available
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
    
    # Check if all required components were found
    if(MKL_INCLUDE_DIR AND MKL_CORE_LIBRARY AND MKL_INTEL_LP64_LIBRARY AND MKL_LINK_THREADING_LIBS)
        set(MKL_FOUND TRUE)
        # Use proper linking flags for non-MSVC compilers
        if(NOT MSVC)
            set(MKL_LIBRARIES -Wl,--no-as-needed ${MKL_INTEL_LP64_LIBRARY} ${MKL_LINK_THREADING_LIBS} ${MKL_CORE_LIBRARY} -Wl,--as-needed)
        else()
            set(MKL_LIBRARIES ${MKL_INTEL_LP64_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_LINK_THREADING_LIBS})
        endif()
        set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
        message(STATUS "Found Intel MKL with detailed linking")
        message(STATUS "MKL Include: ${MKL_INCLUDE_DIRS}")
        message(STATUS "MKL Libraries: ${MKL_LIBRARIES}")
    endif()
endif()

# Handle result
if(MKL_FOUND)
    add_compile_definitions(USE_MKL)
    message(STATUS "Intel MKL enabled")
else()
    message(FATAL_ERROR "Intel MKL requested but not found. Please install Intel MKL or set MKLROOT environment variable.")
endif()
