#pragma once

#include "message.hpp"
#include "command_type.hpp"
#include <iostream>

namespace tpipeline {

/**
 * @brief Example message handler that demonstrates how to process different message types
 */
template <typename T = float>
class MessageHandler {
public:
    static void process_message(const Message<T>& message) {
        std::cout << "Processing message of type: " << static_cast<int>(message.command_type) 
                  << " from sender: " << message.sender_id << std::endl;
        
        switch (message.command_type) {
            case CommandType::FORWARD_TASK:
            case CommandType::BACKWARD_TASK: {
                if (message.template has_payload<Task<T>>()) {
                    auto task = message.template get_payload<Task<T>>();
                    std::cout << "  Task micro_batch_id: " << task.micro_batch_id << std::endl;
                }
                break;
            }
            
            case CommandType::STATUS_REQUEST: {
                std::cout << "  Status request received" << std::endl;
                break;
            }
            
            case CommandType::STATUS_RESPONSE: {
                if (message.template has_payload<StatusInfo>()) {
                    auto status = message.template get_payload<StatusInfo>();
                    std::cout << "  Status: busy=" << status.is_busy 
                              << ", queue_size=" << status.queue_size 
                              << ", info=" << status.additional_info << std::endl;
                }
                break;
            }
            
            case CommandType::ERROR_REPORT: {
                if (message.template has_payload<ErrorInfo>()) {
                    auto error = message.template get_payload<ErrorInfo>();
                    std::cout << "  Error from " << error.stage_name 
                              << ": " << error.error_message 
                              << " (code: " << error.error_code << ")" << std::endl;
                }
                break;
            }
            
            case CommandType::START_TRAINING: {
                std::cout << "  Start training command" << std::endl;
                break;
            }
            
            case CommandType::STOP_TRAINING: {
                std::cout << "  Stop training command" << std::endl;
                break;
            }
            
            case CommandType::HANDSHAKE_REQUEST: {
                if (message.template has_payload<ConfigInfo>()) {
                    auto config = message.template get_payload<ConfigInfo>();
                    std::cout << "  Handshake request for stage: " << config.stage_id << std::endl;
                }
                break;
            }
            
            default: {
                std::cout << "  Unknown message type" << std::endl;
                break;
            }
        }
    }
    
    // Helper function to create common message types
    static Message<T> create_task_message(const Task<T>& task, 
                                         const std::string& sender = "",
                                         const std::string& recipient = "") {
        CommandType cmd_type = (task.type == TaskType::Forward) ? 
                              CommandType::FORWARD_TASK : CommandType::BACKWARD_TASK;
        return Message<T>(cmd_type, task, sender, recipient);
    }
    
    static Message<T> create_status_request(const std::string& sender = "",
                                          const std::string& recipient = "") {
        return Message<T>(CommandType::STATUS_REQUEST, true, sender, recipient);
    }
    
    static Message<T> create_status_response(const StatusInfo& status,
                                           const std::string& sender = "",
                                           const std::string& recipient = "") {
        return Message<T>(CommandType::STATUS_RESPONSE, status, sender, recipient);
    }
    
    static Message<T> create_error_report(const ErrorInfo& error,
                                        const std::string& sender = "",
                                        const std::string& recipient = "") {
        return Message<T>(CommandType::ERROR_REPORT, error, sender, recipient);
    }
};

} // namespace tpipeline
