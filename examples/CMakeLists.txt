# Examples
create_executable(device_manager_example.cpp LIBS tunx_lib)
create_executable(metrics_example.cpp LIBS tunx_lib)

# Network executables
create_executable(tcp_worker.cpp LIBS tunx_lib)
create_executable(tcp_coordinator.cpp LIBS tunx_lib)

create_executable(roce_worker.cpp LIBS tunx_lib)
create_executable(roce_coordinator.cpp LIBS tunx_lib)

# Training executables
create_executable(trainer.cpp LIBS tunx_lib)

# Inference executables
create_executable(inferencer.cpp LIBS tunx_lib)

# Inference executables
create_executable(gpt2_inference.cpp LIBS tunx_lib)

# Test executables
create_executable(test_del.cpp LIBS tunx_device)
create_executable(graph_api_v2_test.cpp LIBS tunx_lib)