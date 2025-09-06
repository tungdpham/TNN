set(CMAKE_SYSTEM_NAME Windows)

# Specify the compiler paths
# Replace with the actual paths to your GCC and G++ executables
# If they are in your PATH, you can just use "gcc.exe" and "g++.exe"
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")

# Tell CMake to use the MinGW Makefiles generator
set(CMAKE_GENERATOR "MinGW Makefiles" CACHE STRING "Force MinGW Makefiles generator")