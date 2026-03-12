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

# Training executables
create_executable(trainer.cpp LIBS tnn_nn)
create_executable(legacy_trainer.cpp LIBS tnn_nn)

# Inference executables
create_executable(inferencer.cpp LIBS tnn_nn)

# Inference executables
create_executable(gpt2_inference.cpp LIBS tnn_nn)

# Test executables
create_executable(graph_test.cpp LIBS tnn_nn)
create_executable(test_del.cpp LIBS tnn_device)