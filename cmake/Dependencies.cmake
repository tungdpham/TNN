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

# zstandard compression library
FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.7
    SOURCE_SUBDIR build/cmake
)

# Make the dependencies available
FetchContent_MakeAvailable(asio nlohmann_json zstd)

# Add global compile definitions for ASIO
add_compile_definitions(ASIO_STANDALONE)

# Windows networking libraries for ASIO
if(WIN32)
    set(WINDOWS_LIBS ws2_32 wsock32 mswsock)
    if(MINGW)
        list(APPEND WINDOWS_LIBS iphlpapi)
    endif()
endif()
