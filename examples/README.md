## Notes
If you plan to use the trainers, then to change the parameters, modify this section:
```c
train_classification_model(
        model, train_loader, test_loader,
        {cifar10_constants::EPOCHS, cifar10_constants::BATCH_SIZE,
         cifar10_constants::LR_DECAY_FACTOR, cifar10_constants::LR_DECAY_INTERVAL,
         cifar10_constants::PROGRESS_PRINT_INTERVAL, DEFAULT_NUM_THREADS, ProfilerType::NORMAL});
```
- ProfilerType currently have 3 Modes: None, Normal (per batch), Cummulative (accumulate throughout the epoch)
- Progress print interval is how many batch per print.
- LR decay interval is how many epochs per decay (exponential)
