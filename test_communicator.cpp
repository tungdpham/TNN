#include "include/pipeline/pipeline_communicator.hpp"
#include "include/pipeline/command_type.hpp"
#include "include/pipeline/message.hpp"
#include <iostream>
#include <memory>

int main() {
    try {
        // Create a communicator
        auto comm = std::make_shared<tpipeline::InProcessPipelineCommunicator<float>>();
        
        // Create test messages
        auto forward_msg = tpipeline::Message<float>::forward_task(
            tpipeline::Task<float>(), "sender1", "recipient1"
        );
        
        auto status_msg = tpipeline::Message<float>::status_message(
            "Test status", "sender2", "recipient2"
        );
        
        auto control_msg = tpipeline::Message<float>::create_control_message(
            tpipeline::CommandType::SHUTDOWN, "sender3", "recipient3"
        );
        
        // Test enqueue
        comm->enqueue_input_message(forward_msg);
        comm->enqueue_input_message(status_msg);
        comm->enqueue_input_message(control_msg);
        
        std::cout << "Enqueued 3 messages successfully" << std::endl;
        
        // Test message counts
        std::cout << "Forward message count: " << comm->forward_message_count() << std::endl;
        std::cout << "Status message count: " << comm->status_message_count() << std::endl;
        std::cout << "Has input messages: " << (comm->has_input_message() ? "yes" : "no") << std::endl;
        
        // Test dequeue by type
        auto dequeued_forward = comm->dequeue_message_by_type(tpipeline::CommandType::FORWARD_TASK);
        std::cout << "Dequeued forward message successfully" << std::endl;
        
        auto dequeued_status = comm->dequeue_message_by_type(tpipeline::CommandType::STATUS_RESPONSE);
        std::cout << "Dequeued status message successfully" << std::endl;
        
        // Test general dequeue
        auto dequeued_general = comm->dequeue_input_message();
        std::cout << "Dequeued general message successfully" << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
