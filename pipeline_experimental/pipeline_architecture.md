# Intended architecture
The pipelining model should consists of a Coordinator, which will have multiple Stages and each Stage should have ability to retrieve, store, and send Forward/Backward tasks (these tasks includes a Tensor either be the input or the gradient). 

This is because when first stage have finished forwarding batch 2, second stage might not have finished forwarding batch 1, so queues are needed. Note that each stage should only be processing only 1 task, more would be meaningless and cause more overhead.

The initial idea was to have each stage runs these things in parallel:
+ Task Retrieval 
+ Process Task
+ Task Sending

However, this turns out to be quite a terrible idea as each of these tasks needs a loop if parallelized, causing massive resource waste most of the time. This is where our next idea comes in:

Event-based Task processing:
- For each stage, a main event will listen for incoming tasks, which will be done via its communicator. When it receives a task, the main event will spawns a thread which will process task and when it finishes processing, it will send the output task to the corresponding recipient. When the send succeeds, automatically closes the thread but does not terminate it (for thread pool re usage).

# Example
If we split the model into 4 stages and each mini-batch is split into 4 microbatches, the pipelining should work as follows:

## Initialization:
The Pipeline Coordinator is told to train on batch i-th, say any arbitrary batch. The coordinator splits the batch into 4 microbatches that is ready to be forwarded by the stages.

## Forward Phrase
- The coordinator uses Communicator to enqueue the 4 microbatch into Stage 1's pending forward task queue. 
- The states of each stage should looks like the following:
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


And you see the pattern, the task continue so on. An important note: After finishing forwarding any microbatch, the last stage, here is stage 4, will send its resulting Tensor to the coordinator for immediate Loss computation.

## Compute Gradient Phrase
- Now when the coordinator has computed the Loss for every microbatch's, it will start computing the gradient with respect to the Loss using the Loss class (which should have compute_loss and compute_gradient functions)

# Backward Phrase
- The backward phrase will happen as a normal synchronous, as splitting it into microbatches is provably meaningless in terms of efficiency (computation) and accuracy.
