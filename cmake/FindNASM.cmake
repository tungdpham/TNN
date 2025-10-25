# FindNASM.cmake - Locate the NASM assembler
# This module finds the NASM assembler for use in building assembly code.

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