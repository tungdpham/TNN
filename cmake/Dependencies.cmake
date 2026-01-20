# Third-party dependencies management using FetchContent

include(FetchContent)

# ASIO (header-only library)
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG asio-1-30-2
)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# stb - single-file public domain libraries (stb_image for image loading)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

# spdlog
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0
)

# Configure spdlog to use external fmt library
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

# cuDNN frontend
FetchContent_Declare(
    cudnn_frontend
    GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG main
)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Make the dependencies available
FetchContent_MakeAvailable(asio nlohmann_json googletest stb spdlog cudnn_frontend)

# Add global compile definitions for ASIO
add_compile_definitions(ASIO_STANDALONE)

# Windows networking libraries for ASIO
if(WIN32)
    set(WINDOWS_LIBS ws2_32 wsock32 mswsock)
    if(MINGW)
        list(APPEND WINDOWS_LIBS iphlpapi)
    endif()
endif()
