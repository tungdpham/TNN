# Examples
create_default(hardware_info_example hardware_info_example.cpp)
create_default(device_manager_example device_manager_example.cpp)

# Network executables
create_default(tcp_worker tcp_worker.cpp)
create_default(tcp_coordinator tcp_coordinator.cpp)

create_default(roce_worker roce_worker.cpp)
create_default(roce_coordinator roce_coordinator.cpp)

# Inspection executables
create_default(inspect_openwebtext inspect_openwebtext.cpp)

# Training executables
create_default(trainer trainer.cpp)

# Inference executables
create_default(gpt2_inference gpt2_inference.cpp)

# Test executables
create_default(fp16_test fp16_test.cpp)
create_default(correctness_test correctness_test.cpp)
create_default(sequential_test sequential_test.cpp)