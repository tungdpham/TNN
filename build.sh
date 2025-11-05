#!/bin/bash

# Build script for TNN project

set -e 

# convenient colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# default vars
BUILD_TYPE="Release"
ENABLE_OPENMP=OFF
ENABLE_CUDA=OFF
ENABLE_TBB=ON
ENABLE_MKL=OFF
ENABLE_DEBUG=OFF
CLEAN_BUILD=false
VERBOSE=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -d, --debug         Enable debug build with sanitizers"
    echo "  -v, --verbose       Enable verbose build output"
    echo "  --tbb               Enable Intel TBB support (on by default)"
    echo "  --openmp            Enable OpenMP support"
    echo "  --mkl               Enable Intel MKL support (off by default)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build with default settings"
    echo "  $0 --clean          # Clean build"
    echo "  $0 --debug          # Debug build with sanitizers"
    echo "  $0 --tbb            # Enable Intel TBB support (already on by default)"
    echo "  $0 --openmp         # Enable OpenMP support"
}

# parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            ENABLE_DEBUG=ON
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --tbb)
            ENABLE_TBB=ON
            shift
            ;;
        --openmp)
            ENABLE_OPENMP=ON #if we want both openmp and tbb
            shift
            ;;
        --mkl)
            ENABLE_MKL=ON
            shift
            ;;
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# print build configuration
echo -e "${GREEN}TNN CMake Build Configuration:${NC}"
echo "  Build Type: $BUILD_TYPE"
echo "  OpenMP: $ENABLE_OPENMP"
echo "  Intel TBB: $ENABLE_TBB"
echo "  Debug Mode: $ENABLE_DEBUG"
echo ""

# clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning all build artifacts.${NC}"
    find . -name "CMakeFiles" -type d -exec rm -rf {} +
    find . -name "cmake_install.cmake" -type f -delete
    find . -name "CMakeCache.txt" -type f -delete
    find . -name "Makefile" -type f -delete
    rm -rf bin/ lib/ build/ compile_commands.json
    
    echo "Cleaned build files from current directory and all subdirectories"
    echo ""
fi

# configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENABLE_OPENMP="$ENABLE_OPENMP"
    -DENABLE_TBB="$ENABLE_TBB"
    -DENABLE_DEBUG="$ENABLE_DEBUG"
    -DENABLE_MKL="$ENABLE_MKL"
)

cmake . "${CMAKE_ARGS[@]}"

# doing full build
echo -e "${GREEN}Building project...${NC}"
if [ "$VERBOSE" = true ]; then
    cmake --build . --verbose
else
    cmake --build . -j$(nproc)
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Available executables in bin/:${NC}"
echo "  - mnist_cnn_trainer"
echo "  - cifar10_cnn_trainer"
echo "  - cifar100_cnn_trainer"
echo "  - uji_ips_trainer"
echo "  - mnist_cnn_test"
echo "  - pipeline_test"
echo "  - network_worker"
echo "  - distributed_pipeline_docker"
echo "  - More because I'm lazy to type them all out"
echo ""
echo -e "${YELLOW}To run a specific executable:${NC}"
echo "  ./bin/mnist_cnn_trainer or ./bin/Release/mnist_cnn_trainer (Windows with MSVC)"

