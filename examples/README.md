## Configuring Exposed Parameters

If you plan to use the trainers, to to change the parameters, create a .env in root directory and change the params there.

See .env.example in root directory for available tunable parameters. For more in-depth tuning, you need to change the code.

## Running Coordinator and Network Workers

To use coordinator, you can run semi_async_pipeline_coordinator. But first, you need to configure host and port of expected worker endpoints in .env in root dir. After that, run network workers before coordinator. Then simply run coordinator executable by:

```bash
# Run network worker on port 8001
./bin/network_worker 8001
```

Then, run coordinator after all worker:

```bash
# Run semi async coordinator
./bin/semi_async_pipeline_coordinator
```