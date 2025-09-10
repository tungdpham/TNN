# Architecture
For now, the pipelining model consists of a Coordinator, multiple Stages. For modularity, the Coordinator and the Stages should only communicate via the Communicator, which have the ability to retrieve, store, and send messages (these messages comes with a metadata about the request type). The Stages and the Coordinator does not know the method which the Communicator uses to communicate, they know these methods exists and call them. Allows nice scalability for different communication methods/frameworks. 

Each stage will have a input message queue. This is because when first Stage have finished forwarding batch 2, second Stage might not have finished forwarding batch 1. Note that each Stage should only be processing only 1 task, more would be meaningless and will cause unneccesary overhead.

## Event-based Task processing:
- For each stage, their Communicator will have an I/O thread that will listen for incoming messages. 
- On retrieval of a message, the I/O will decompress, deserialize the message and invoke a callback that notify the main event loop to wake up.
- When active, the event loop polls messages from the pending message queue, and start processing them until none is left. 
- On each successful message processing, the output will be flushed to a pre-defined recipient, which will be next stage for forward, previous stage for backward, or coordinator by default. 
- An error will be reported to the coordinator if there were corrupted messages or error during message processing.
- When the message queue is empty, the main event loop will go back to idle/sleep state.

## Network auto-distribution
- For network distribution, user will only need to specify the config of the whole model, the number of stages, and number of microbatches, from which the coordinator is created. 
- To do this, the coordinator needs to do an initial handshake with all the specified endpoints that the user specifies, which are the machines we will distribute the model. 
- Once it verifies the connection, it serialize each stage into config and send it over the network. Meanwhile, the recipient, being a machine, will listen for messages and create the model from that config. 
- For each machine, once it has successfully created the Pipeline Stage from given config, it will then ping back to the coordinator that it's ready. 
- Once all the Stages are ready, the coordinator will start forwarding inputs. The rest of the training loop will be as below.

# Training
There will be 3 types of training: synchronous (normal training), semi-asynchronous (immediate backwards), asynchronous (no-wait, susceptible to stale gradients).

## Synchronous Training
If we split the model into 4 stages and each mini-batch is split into 4 microbatches, the pipelining should work as follows:

### Initialization:
The Pipeline Coordinator is told to train on batch i-th, say any arbitrary batch. The coordinator splits the batch into 4 microbatches that is ready to be forwarded by the stages.

### Forward Phrase
The coordinator uses Communicator to enqueue the 4 microbatch into Stage 1's pending forward task queue. 

The states of each stage should looks like the following:
    + Stage 1: Current task: empty, Current Queue: Forward microbatch 1,2,3,4
    + Stage 2: Current task: empty, Current Queue: empty 
    + Stage 3: Current task: empty, Current Queue: empty
    + Stage 4: Current task: empty, Current Queue: empty

- As Stage 1 is free, it checks its forward tasks and see that there is a pending task and starts processing it.
- The states of each stage should looks like the following:
    + Stage 1: Current task: Forward microbatch 1, Current Queue: Forward microbatch 2,3,4
    + Stage 2: Current task: empty, Current Queue: empty 
    + Stage 3: Current task: empty, Current Queue: empty
    + Stage 4: Current task: empty, Current Queue: empty

- After Stage 1 finish forwarding microbatch 1, it uses its communicator to send its result Tensor to Stage 2's forward task queue.
- The states of each stage should looks like the following:
    + Stage 1: Current task: Forward microbatch 2, Current Queue: Forward microbatch 2,3,4
    + Stage 2: Current task: empty, Current Queue: Forward microbatch 1
    + Stage 3: Current task: empty, Current Queue: empty
    + Stage 4: Current task: empty, Current Queue: empty

- Stage 2 checks and decides to forward microbatch 1. Stage 1 continues to check that there is still a pending task, and forwards microbatch 2.
- The states of each stage should looks like the following:
    + Stage 1: Current task: Forward microbatch 2, Current Queue: Forward microbatch 3,4
    + Stage 2: Current task: Forward microbatch 1, Current Queue: empty
    + Stage 3: Current task: empty, Current Queue: empty
    + Stage 4: Current task: empty, Current Queue: empty

- Now when Stage 1 has finished forwarding microbatch 2, but if for any reasons Stage 2 hasn't finished forwarding microbatch 1, the microbatch 2 will be enqueued. Stage 1 then checks and forward microbatch 3.
- The states of each stage should looks like the following:
    + Stage 1: Current task: Forward microbatch 3, Current Queue: Forward microbatch 4
    + Stage 2: Current task: Forward microbatch 1, Current Queue: Forward microbatch 2
    + Stage 3: Current task: empty, Current Queue: empty
    + Stage 4: Current task: empty, Current Queue: empty

And you see the pattern, the task continue so on. An important note: After finishing forwarding any microbatch, the last stage, here is stage 4, will send its resulting Tensor to the coordinator.

### Compute Gradient Phrase
- Now when the coordinator has computed the Loss for every microbatch's, it will start computing the gradient with respect to the Loss using the Loss class (which should have compute_loss and compute_gradient functions)

# Backward Phrase
- The backward phrase will do the same but backwards.


## Semi-Asynchronous Training
For semi-asynchronous training, it will happen the same as synchronous except each microbatch that get to the coordinator will have immediate loss computation and be backwarded.