# Compiler and CUDA settings
CXX = g++
NVCC = nvcc

# Feature flags
ENABLE_OPENMP ?= 1
ENABLE_CUDA ?= 0
ENABLE_BLAS ?= 0

# Source files
CXX_SOURCES = $(wildcard matrix/*.cpp neural/*.cpp utils/*.cpp)
HEADERS = $(wildcard matrix/*.h neural/*.h utils/*.h matrix/*.hpp layers/*.hpp)
CU_SOURCES = $(wildcard matrix/*.cu neural/*.cu utils/*.cu)

# Object files
CXX_OBJ = $(CXX_SOURCES:.cpp=.o)
ifeq ($(ENABLE_CUDA), 1)
	CU_OBJ = $(CU_SOURCES:.cu=.o)
	OBJ = $(CXX_OBJ) $(CU_OBJ)
else
	OBJ = $(CXX_OBJ)
endif

# Compilation flags
CXXFLAGS = -std=c++20 -Wpedantic -O3 -march=x86-64-v3 -flto
NVCCFLAGS = -std=c++20 -O3 -arch=sm_89 --compiler-options -fPIC
LDFLAGS = -lm
CUDA_LDFLAGS = -lm -lcudart -lcublas -lcurand

# Add OpenMP support
ifeq ($(ENABLE_OPENMP), 1)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
	# CUDA_LDFLAGS += -fopenmp
endif

# Add CUDA support
ifeq ($(ENABLE_CUDA), 1)
	CXXFLAGS += -DCUDA_ENABLED
	NVCCFLAGS += -DCUDA_ENABLED
	NVCC_EXISTS := $(shell command -v $(NVCC) 2> /dev/null)
	ifeq ($(NVCC_EXISTS),)
		$(error NVCC not found. Please install CUDA toolkit.)
	endif
endif

# Add BLAS support
ifeq ($(ENABLE_BLAS), 1)
	# Check for OpenBLAS with pkg-config
	ifeq ($(shell pkg-config --exists openblas && echo 1), 1)
		CXXFLAGS += -DUSE_OPENBLAS
		CXXFLAGS += $(shell pkg-config --cflags openblas)
		LDFLAGS += $(shell pkg-config --libs openblas)
	endif
endif

MAIN = main

# Default target
main: main.cpp ${OBJ}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} $^ -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${CXXFLAGS} $^ -o $@ ${LDFLAGS}
endif

# C++ compilation rules
%.o: %.cpp ${HEADERS}
	${CXX} ${CXXFLAGS} -c $< -o $@

# CUDA compilation rules
ifeq ($(ENABLE_CUDA), 1)
%.o: %.cu ${HEADERS}
	${NVCC} ${NVCCFLAGS} -c $< -o $@
endif

# Clean target
clean:
ifeq ($(OS),Windows_NT)
	if exist matrix\*.o del matrix\*.o
	if exist neural\*.o del neural\*.o  
	if exist utils\*.o del utils\*.o
	if exist *.o del *.o
	if exist ${MAIN}.exe del ${MAIN}.exe
	for %%f in ($(TEST_PROGRAMS)) do if exist %%f.exe del %%f.exe
else
	rm -f matrix/*.o neural/*.o utils/*.o layers/*.o *.o ${MAIN}
	rm -f main mnist_trainer mnist_cnn_trainer
	rm -f $(TEST_PROGRAMS)
	rm -f test_activations integration_test
endif

# Test targets
TEST_SOURCES = $(wildcard unit_tests/*.cpp)
TEST_PROGRAMS = $(TEST_SOURCES:unit_tests/%.cpp=%)
TEST_CXXFLAGS = $(CXXFLAGS) -I.

# Generic test compilation rule
%: unit_tests/%.cpp ${HEADERS}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} -I. $< -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${TEST_CXXFLAGS} $< -o $@ ${LDFLAGS}
endif

# MNIST trainer target
mnist_trainer: mnist_trainer.cpp ${HEADERS}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} -I. $< -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${TEST_CXXFLAGS} $< -o $@ ${LDFLAGS}
endif

# MNIST CNN trainer target
mnist_cnn_trainer: mnist_cnn_trainer.cpp ${HEADERS}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} -I. $< -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${TEST_CXXFLAGS} $< -o $@ ${LDFLAGS}
endif

# MNIST CNN test target
mnist_cnn_test: mnist_cnn_test.cpp ${HEADERS}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} -I. $< -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${TEST_CXXFLAGS} $< -o $@ ${LDFLAGS}
endif

# MNIST CNN pipeline trainer target
mnist_cnn_pipeline_trainer: mnist_cnn_pipeline_trainer.cpp ${HEADERS}
ifeq ($(ENABLE_CUDA), 1)
	${NVCC} ${NVCCFLAGS} -I. $< -o $@ ${CUDA_LDFLAGS}
else
	${CXX} ${TEST_CXXFLAGS} $< -o $@ ${LDFLAGS}
endif	

# Build all tests
tests: $(TEST_PROGRAMS)

# Run all tests
run_tests: tests
	@echo "Running all tests..."
	@for test in $(TEST_PROGRAMS); do \
		echo "Running $$test..."; \
		./$$test; \
		echo ""; \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  main           - Build main MNIST program with current settings"
	@echo "  mnist_trainer  - Build MNIST neural network trainer"
	@echo "  mnist_cnn_trainer - Build MNIST CNN tensor neural network trainer"
	@echo "  tests          - Build all test programs"
	@echo "  clean          - Remove object files and executables"
	@echo ""
	@echo "Environment variables:"
	@echo "  ENABLE_OPENMP  - Enable OpenMP (default: 1)"
	@echo "  ENABLE_CUDA    - Enable CUDA (default: 0)"

.PHONY: main clean help tests run_tests mnist_trainer mnist_cnn_trainer mnist_cnn_test mnist_cnn_pipeline_trainer $(TEST_PROGRAMS)
