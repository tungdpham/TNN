message(STATUS "Finding Intel oneDNN (DNNL)")

if(NOT DEFINED dnnl_ROOT)
    if(DEFINED ENV{DNNLROOT})
        set(dnnl_ROOT "$ENV{DNNLROOT}")
    elseif(EXISTS "/opt/intel/oneapi/dnnl/latest/cpu_tbb")
        set(dnnl_ROOT "/opt/intel/oneapi/dnnl/latest/cpu_tbb")
    endif()
endif()

find_package(dnnl CONFIG QUIET)
if(dnnl_FOUND)
    set(DNNL_FOUND TRUE)
    message(STATUS "Found Intel oneDNN via CMake config (DNNL::dnnl)")
else()
    # Fall back to manual search
    find_path(DNNL_INCLUDE_DIR
        NAMES dnnl.hpp dnnl.h
        PATHS
            /opt/intel/oneapi/dnnl/latest/cpu_tbb/include
            /opt/intel/oneapi/dnnl/latest/include
            $ENV{DNNLROOT}/include
            $ENV{MKLROOT}/../dnnl/latest/include
            /usr/local/include
            /usr/include
    )

    find_library(DNNL_LIBRARY
        NAMES dnnl
        PATHS
            /opt/intel/oneapi/dnnl/latest/cpu_tbb/lib
            /opt/intel/oneapi/dnnl/latest/lib
            $ENV{DNNLROOT}/lib
            $ENV{MKLROOT}/../dnnl/latest/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib
    )

    if(DNNL_INCLUDE_DIR AND DNNL_LIBRARY)
        set(DNNL_FOUND TRUE)
        set(DNNL_INCLUDE_DIRS ${DNNL_INCLUDE_DIR})
        set(DNNL_LIBRARIES ${DNNL_LIBRARY})
        message(STATUS "Found Intel oneDNN (manual search)")
        message(STATUS "oneDNN Include: ${DNNL_INCLUDE_DIRS}")
        message(STATUS "oneDNN Libraries: ${DNNL_LIBRARIES}")
    endif()
endif()

if(DNNL_FOUND)
    add_compile_definitions(USE_DNNL)
    message(STATUS "Intel oneDNN (DNNL) enabled")

    set(DNNL_INTEL_RUNTIME_LIBS "")
    set(DNNL_INTEL_RUNTIME_LIB_DIR "")

    find_library(DNNL_SYCL_LIB
        NAMES sycl sycl8 sycl-preview
        PATHS
            /opt/intel/oneapi/compiler/latest/lib
            /opt/intel/oneapi/compiler/latest/linux/lib
            /opt/intel/oneapi/2025.3/lib
            /opt/intel/oneapi/compiler/2025.3/lib
            /opt/intel/oneapi/compiler/2025.2/lib
            /opt/intel/oneapi/compiler/2025.1/lib
            /opt/intel/oneapi/compiler/2024.2/lib
            $ENV{CMPLR_ROOT}/lib
        NO_DEFAULT_PATH
    )

    if(DNNL_SYCL_LIB)
        get_filename_component(DNNL_INTEL_RUNTIME_LIB_DIR "${DNNL_SYCL_LIB}" DIRECTORY)
        message(STATUS "Found Intel compiler runtime dir: ${DNNL_INTEL_RUNTIME_LIB_DIR}")

        foreach(_ilib sycl svml imf intlc irng)
            find_library(_dnnl_rt_${_ilib}
                NAMES ${_ilib}
                HINTS "${DNNL_INTEL_RUNTIME_LIB_DIR}"
                NO_DEFAULT_PATH
            )
            if(_dnnl_rt_${_ilib})
                list(APPEND DNNL_INTEL_RUNTIME_LIBS "${_dnnl_rt_${_ilib}}")
                message(STATUS "  Found Intel runtime lib: ${_dnnl_rt_${_ilib}}")
            endif()
        endforeach()
    else()
        message(STATUS "Intel compiler runtime (libsycl etc.) not found; "
            "linking may fail if the installed libdnnl.so is the DPC++ variant. "
            "Set CMPLR_ROOT to your Intel compiler installation if needed.")
    endif()
else()
    message(FATAL_ERROR "Intel oneDNN requested but not found. "
        "Install via 'apt install libdnnl-dev', build from source "
        "(https://github.com/oneapi-src/oneDNN), Intel oneAPI toolkit, "
        "or set DNNLROOT environment variable.")
endif()
