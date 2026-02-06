# Examples
create_executable(hardware_info_example.cpp LIBS tnn_nn)
create_executable(device_manager_example.cpp LIBS tnn_nn)

# Network executables
create_executable(tcp_worker.cpp LIBS tnn_lib)
create_executable(tcp_coordinator.cpp LIBS tnn_lib)

create_executable(roce_worker.cpp LIBS tnn_lib)
create_executable(roce_coordinator.cpp LIBS tnn_lib)

create_executable(legacy_tcp_coordinator.cpp LIBS tnn_lib)
create_executable(legacy_roce_coordinator.cpp LIBS tnn_lib)

# Inspection executables
create_executable(inspect_openwebtext.cpp LIBS tnn_nn)

# Training executables
create_executable(trainer.cpp LIBS tnn_nn)
create_executable(fp16_trainer.cpp LIBS tnn_nn)
create_executable(bf16_trainer.cpp LIBS tnn_nn)

create_executable(legacy_trainer.cpp LIBS tnn_nn)

# Inference executables
create_executable(inferencer.cpp LIBS tnn_nn)

# Inference executables
create_executable(gpt2_inference.cpp LIBS tnn_nn)

# Test executables
create_executable(fp16_test.cpp LIBS tnn_nn)
create_executable(bf16_test.cpp LIBS tnn_nn)
create_executable(correctness_test.cpp LIBS tnn_nn)
create_executable(sequential_test.cpp LIBS tnn_nn)
create_executable(tensor_usage_test.cpp LIBS tnn_nn)
create_executable(dptr_test.cpp LIBS tnn_nn)