# Build script for DCNN project

# Default values
$BUILD_TYPE = "Release"
$BUILD_DIR = "."
$ENABLE_OPENMP = $false
$ENABLE_CUDA = $false
$ENABLE_TBB = $true
$ENABLE_DEBUG = $false
$CLEAN_BUILD = $false
$VERBOSE = $false

# Define colors for output (PowerShell equivalent)
function Write-Color($text, $color) {
    Write-Host -ForegroundColor $color $text
}

# Help function
function Show-Help {
    Write-Host "Usage: .\build.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "    -h, --help          Show this help message"
    Write-Host "    -c, --clean         Clean build directory before building"
    Write-Host "    -d, --debug         Enable debug build with sanitizers"
    Write-Host "    -v, --verbose       Enable verbose build output"
    Write-Host "    --cuda              Enable CUDA support"
    Write-Host "    --tbb               Enable Intel TBB support (on by default)"
    Write-Host "    --openmp            Enable OpenMP support"
    Write-Host "    --build-dir DIR     Set custom build directory (default: . [current dir])"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "    .\build.ps1             # Build with default settings"
    Write-Host "    .\build.ps1 --clean     # Clean build"
    Write-Host "    .\build.ps1 --debug     # Debug build with sanitizers"
    Write-Host "    .\build.ps1 --cuda      # Enable CUDA support"
    Write-Host "    .\build.ps1 --tbb       # Enable Intel TBB support"
    Write-Host "    .\build.ps1 --openmp    # Enable OpenMP support"
}

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = $args[$i]
    switch ($arg) {
        "-h" { Show-Help; exit 0 }
        "--help" { Show-Help; exit 0 }
        "-c" { $CLEAN_BUILD = $true }
        "--clean" { $CLEAN_BUILD = $true }
        "-d" { $BUILD_TYPE = "Debug"; $ENABLE_DEBUG = $true }
        "--debug" { $BUILD_TYPE = "Debug"; $ENABLE_DEBUG = $true }
        "-v" { $VERBOSE = $true }
        "--verbose" { $VERBOSE = $true }
        "--cuda" { $ENABLE_CUDA = $true }
        "--tbb" { $ENABLE_TBB = $true }
        "--openmp" { $ENABLE_OPENMP = $true }
        "--build-dir" {
            if ($i + 1 -lt $args.Count) {
                $i++
                $BUILD_DIR = $args[$i]
            } else {
                Write-Color "Error: Missing value for --build-dir" "Red"
                Show-Help
                exit 1
            }
        }
        Default {
            Write-Color "Unknown option: $arg" "Red"
            Show-Help
            exit 1
        }
    }
}

# Print build configuration
Write-Color "DCNN CMake Build Configuration:" "Green"
Write-Host "  Build Type: $BUILD_TYPE"
Write-Host "  Build Directory: $BUILD_DIR"
Write-Host "  OpenMP: $ENABLE_OPENMP"
Write-Host "  CUDA: $ENABLE_CUDA"
Write-Host "  Intel TBB: $ENABLE_TBB"
Write-Host "  Debug Mode: $ENABLE_DEBUG"
Write-Host ""

# Clean build directory if requested
if ($CLEAN_BUILD) {
    Write-Color "Cleaning build artifacts..." "Yellow"
    if ($BUILD_DIR -eq ".") {
        # Clean only build artifacts from current directory
        Remove-Item CMakeCache.txt, CMakeFiles, Makefile, cmake_install.cmake -Force -Recurse -ErrorAction SilentlyContinue
        Remove-Item bin, lib, compile_commands.json -Force -Recurse -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Include "*.vcxproj", "*.vcxproj.filters", "*.vcxproj.user" | Remove-Item -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Directory -Name "*.dir" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item *.sln, x64, Debug, Release, *.log -Force -Recurse -ErrorAction SilentlyContinue
        
        Write-Host "Cleaned build files from current directory"
    } else {
        # Clean or recreate build directory
        if (Test-Path $BUILD_DIR) {
            Remove-Item -Path $BUILD_DIR -Recurse -Force
            Write-Host "Removed build directory: $BUILD_DIR"
        }
        New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
        Write-Host "Created fresh build directory: $BUILD_DIR"
    }
    Write-Host ""
}

# Create build directory if it's not the current directory
if ($BUILD_DIR -ne ".") {
    if (-not (Test-Path $BUILD_DIR)) {
        New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    }
    Set-Location $BUILD_DIR
}

# Configure with CMake
Write-Color "Configuring with CMake..." "Green"
$CMAKE_ARGS = @(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DENABLE_OPENMP=$ENABLE_OPENMP"
    "-DENABLE_CUDA=$ENABLE_CUDA"
    "-DENABLE_TBB=$ENABLE_TBB"
    "-DENABLE_DEBUG=$ENABLE_DEBUG"
    "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
)

if ($BUILD_DIR -eq ".") {
    cmake . @CMAKE_ARGS
} else {
    cmake .. @CMAKE_ARGS
    $INSTALL_PREFIX = (Get-Item ..).FullName
    $CMAKE_ARGS += "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    cmake . @CMAKE_ARGS
}

# Build
Write-Color "Building project..." "Green"
if ($VERBOSE) {
    cmake --build . --config $BUILD_TYPE --verbose
} else {
    # Get number of cores for multi-threaded build
    $cores = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
    cmake --build . --config $BUILD_TYPE "-j$cores"
}

Write-Color "Build completed successfully!" "Green"
Write-Host ""
Write-Color "Available executables in bin/:" "Yellow"
Write-Host "  - mnist_cnn_trainer"
Write-Host "  - cifar10_cnn_trainer"
Write-Host "  - cifar100_cnn_trainer"
Write-Host "  - uji_ips_trainer"
Write-Host "  - mnist_cnn_test"
Write-Host "  - pipeline_test"
Write-Host "  - network_worker"
Write-Host "  - distributed_pipeline_docker"
Write-Host "  - More because I'm lazy to type them all out"
Write-Host ""
Write-Color "To run a specific executable:" "Yellow"
Write-Host "  cd $BUILD_DIR && .\bin\mnist_cnn_trainer.exe"
Write-Host ""
Write-Color "To run tests:" "Yellow"
Write-Host "  cd $BUILD_DIR && make run_tests"