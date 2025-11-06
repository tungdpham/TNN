# Compiler-specific flags for different platforms and build types

if(MSVC)
    message(STATUS "Configuring MSVC compiler flags")
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
    message(STATUS "Configuring MinGW compiler flags")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
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
    message(STATUS "Configuring GCC/Clang compiler flags")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto=auto -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fverbose-asm -march=native")
    add_compile_options(
        -Wall 
        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wpedantic>
    )
    if(ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:
                -save-temps=obj -masm=intel -fno-lto
            >
        )
    endif()
endif()
