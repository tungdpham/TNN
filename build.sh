#!/bin/bash

# Build script for DCNN project

set -e 

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
BUILD_TYPE="Release"
BUILD_DIR="."
ENABLE_OPENMP=ON
ENABLE_CUDA=OFF
ENABLE_TBB=OFF
ENABLE_DEBUG=OFF
CLEAN_BUILD=false
VERBOSE=false

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -d, --debug         Enable debug build with sanitizers"
    echo "  -v, --verbose       Enable verbose build output"
    echo "  --cuda              Enable CUDA support"
    echo "  --tbb               Enable Intel TBB support"
    echo "  --no-openmp         Disable OpenMP support"
    echo "  --build-dir DIR     Set custom build directory (default: . [current dir])"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build with default settings"
    echo "  $0 --clean          # Clean build"
    echo "  $0 --debug          # Debug build with sanitizers"
    echo "  $0 --cuda           # Enable CUDA support"
    echo "  $0 --tbb            # Enable Intel TBB support"
}

# Parse command line arguments
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
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --tbb)
            ENABLE_TBB=ON
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP=OFF
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Print build configuration
echo -e "${GREEN}DCNN CMake Build Configuration:${NC}"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Directory: $BUILD_DIR"
echo "  OpenMP: $ENABLE_OPENMP"
echo "  CUDA: $ENABLE_CUDA"
echo "  Intel TBB: $ENABLE_TBB"
echo "  Debug Mode: $ENABLE_DEBUG"
echo ""

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    if [ "$BUILD_DIR" = "." ]; then
        # Clean only build artifacts, not source files
        rm -f cmake_install.cmake CMakeCache.txt Makefile
        rm -rf CMakeFiles/
        rm -f mnist_cnn_trainer cifar10_cnn_trainer cifar100_cnn_trainer uji_ips_trainer
        rm -f mnist_cnn_test pipeline_test network_worker distributed_pipeline_docker
    else
        rm -rf "$BUILD_DIR"
    fi
fi

# Create build directory only if it's not current directory
if [ "$BUILD_DIR" != "." ]; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENABLE_OPENMP="$ENABLE_OPENMP"
    -DENABLE_CUDA="$ENABLE_CUDA"
    -DENABLE_TBB="$ENABLE_TBB"
    -DENABLE_DEBUG="$ENABLE_DEBUG"
)

if [ "$BUILD_DIR" = "." ]; then
    cmake . "${CMAKE_ARGS[@]}"
else
    cmake .. "${CMAKE_ARGS[@]}"
    
    # Ensure the generated install script uses an absolute, correct install prefix
    # (project root). Prefer realpath, fallback to a safe pwd-based approach.
    if command -v realpath >/dev/null 2>&1; then
        INSTALL_PREFIX="$(realpath ..)"
    else
        INSTALL_PREFIX="$(cd .. && pwd)"
    fi
    CMAKE_ARGS+=( -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" )
    
    cmake . "${CMAKE_ARGS[@]}"
fi

# Build
echo -e "${GREEN}Building project...${NC}"
if [ "$VERBOSE" = true ]; then
    cmake --build . --verbose
else
    cmake --build . -j$(nproc)
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Available executables in $BUILD_DIR/:${NC}"
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
echo "  cd $BUILD_DIR && ./mnist_cnn_trainer"
echo ""
echo -e "${YELLOW}To run tests:${NC}"
echo "  cd $BUILD_DIR && make run_tests"

