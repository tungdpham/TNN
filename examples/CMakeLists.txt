# Examples
create_executable(hardware_info_example.cpp LIBS synet_nn)
create_executable(device_manager_example.cpp LIBS synet_nn)
create_executable(metrics_example.cpp LIBS synet_nn)

# Network executables
create_executable(tcp_worker.cpp LIBS synet_lib)
create_executable(tcp_coordinator.cpp LIBS synet_lib)

create_executable(roce_worker.cpp LIBS synet_lib)
create_executable(roce_coordinator.cpp LIBS synet_lib)

create_executable(legacy_tcp_coordinator.cpp LIBS synet_lib)
create_executable(legacy_roce_coordinator.cpp LIBS synet_lib)

# Training executables
create_executable(trainer.cpp LIBS synet_nn)
create_executable(legacy_trainer.cpp LIBS synet_nn)

# Inference executables
create_executable(inferencer.cpp LIBS synet_nn)

# Inference executables
create_executable(gpt2_inference.cpp LIBS synet_nn)

# Test executables
create_executable(graph_test.cpp LIBS synet_nn)
create_executable(test_del.cpp LIBS synet_device)
create_executable(graph_api_v2_test.cpp LIBS synet_nn)