## Configuring Exposed Parameters

If you plan to use the trainers with environment file, to change the parameters, create a .env in root directory and change the params there.

If you plan to use the trainers or inferencer with json config, create one following examples in the configs directory. More detailed parameter documentation will be available in the docs in the future.

See .env.example in root directory for available tunable parameters. For more in-depth tuning, you need to change the code.

## Running Coordinator and Network Workers

Running the coordinator is similar, but first run the workers, which will receive commands from the coordinator.

```bash
# Run worker over TCP/IPv4 (default)
./bin/tcp_worker 8001

# Run worker with CUDA
./bin/tcp_worker 8001 --gpu 

# Run network worker with custom number of IO threads (default is 4)
./bin/tcp_worker 8001 --io-threads 8

# Run network worker with custom number of worker threads (default is 8)
./bin/tcp_worker 8001 --num-threads 4

# Use multiple options
./bin/tcp_worker 8001 --gpu --io-threads 8 --num-threads 16

# Run worker with RDMA (default)
./bin/roce_worker --port 8001 --device {device_id}
```

Then, run coordinator after all worker:

```bash
# Run coordinator over TCP/IPv4 (default)
./bin/tcp_coordinator

# Run coordinator with RDMA (default)
./bin/roce_coordinator --device {device_id}
```