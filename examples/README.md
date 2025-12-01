## Configuring Exposed Parameters

If you plan to use the trainers, to to change the parameters, create a .env in root directory and change the params there.

See .env.example in root directory for available tunable parameters. For more in-depth tuning, you need to change the code.

## Running Coordinator and Network Workers

To use coordinator, you can run semi_async_pipeline_coordinator. But first, you need to configure host and port of expected worker endpoints in .env in root dir. After that, run network workers before coordinator. Then simply run coordinator executable by:

```bash
# Run network worker on port 8001
./bin/network_worker 8001

# Run network worker with GPU
./bin/network_worker 8001 --gpu 

# Run network worker with custom number of IO threads (default is 4)
./bin/network_worker 8001 --io-threads 8

# Run network worker with custom number of worker threads (default is 8)
./bin/network_worker 8001 --num-threads 4

# Use multiple options
./bin/network_worker 8001 --gpu --io-threads 8 --num-threads 16
```

Then, run coordinator after all worker:

```bash
# Run semi async coordinator
./bin/semi_async_pipeline_coordinator

# Run tiny imagenet coordinator with ResNet-18
./bin/coordinator_tiny_imagenet
```