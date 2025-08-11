#pragma once

#include "command_type.hpp"
#include "task.hpp"
#include <variant>
#include <string>
#include <map>
#include <chrono>

namespace tpipeline {

/**
 * @brief Status information for system state messages
 */
struct StatusInfo {
    bool is_busy = false;
    size_t queue_size = 0;
    std::chrono::steady_clock::time_point timestamp;
    std::string additional_info;
    
    StatusInfo() : timestamp(std::chrono::steady_clock::now()) {}
    StatusInfo(bool busy, size_t queue_sz, const std::string& info = "")
        : is_busy(busy), queue_size(queue_sz), 
          timestamp(std::chrono::steady_clock::now()), 
          additional_info(info) {}
};

/**
 * @brief Error information for error reporting
 */
struct ErrorInfo {
    std::string error_message;
    std::string stage_name;
    int error_code = 0;
    std::chrono::steady_clock::time_point timestamp;
    
    ErrorInfo() : timestamp(std::chrono::steady_clock::now()) {}
    ErrorInfo(const std::string& msg, const std::string& stage = "", int code = 0)
        : error_message(msg), stage_name(stage), error_code(code),
          timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Configuration data for network handshake
 */
struct ConfigInfo {
    std::map<std::string, std::string> parameters;
    std::string model_config_json;
    std::string stage_id;
    
    ConfigInfo() = default;
    ConfigInfo(const std::string& config, const std::string& id)
        : model_config_json(config), stage_id(id) {}
};

/**
 * @brief A unified message structure for all pipeline communication.
 * 
 * This message type can handle all forms of communication between pipeline
 * stages and the coordinator, from task processing to control commands
 * and error reporting.
 */
template <typename T = float>
struct Message {
    CommandType command_type;
    
    // The payload uses std::variant for type-safe, memory-efficient storage
    std::variant<
        Task<T>,        // For FORWARD_TASK, BACKWARD_TASK
        StatusInfo,     // For STATUS_REQUEST, STATUS_RESPONSE, HEALTH_CHECK
        ErrorInfo,      // For ERROR_REPORT, TASK_FAILURE
        ConfigInfo,     // For HANDSHAKE_REQUEST, CONFIG_RECEIVED
        std::string,    // For simple text-based messages
        bool,           // For simple boolean signals (READY_SIGNAL, etc.)
        int             // For numeric data (error codes, counts, etc.)
    > payload;
    
    // Optional metadata
    std::string sender_id;
    std::string recipient_id;
    std::chrono::steady_clock::time_point timestamp;
    
    // Constructors for different payload types
    Message(CommandType cmd_type)
        : command_type(cmd_type), timestamp(std::chrono::steady_clock::now()) {}
    
    template<typename PayloadType>
    Message(CommandType cmd_type, const PayloadType& data)
        : command_type(cmd_type), payload(data), 
          timestamp(std::chrono::steady_clock::now()) {}
    
    template<typename PayloadType>
    Message(CommandType cmd_type, const PayloadType& data, 
            const std::string& sender, const std::string& recipient)
        : command_type(cmd_type), payload(data), 
          sender_id(sender), recipient_id(recipient),
          timestamp(std::chrono::steady_clock::now()) {}
    
    // Helper methods to check payload type
    bool is_task() const {
        return command_type == CommandType::FORWARD_TASK || 
               command_type == CommandType::BACKWARD_TASK;
    }
    
    bool is_control_command() const {
        return command_type == CommandType::START_TRAINING ||
               command_type == CommandType::STOP_TRAINING ||
               command_type == CommandType::PAUSE_TRAINING ||
               command_type == CommandType::RESUME_TRAINING;
    }
    
    bool is_error() const {
        return command_type == CommandType::ERROR_REPORT ||
               command_type == CommandType::TASK_FAILURE;
    }
    
    // Helper methods to safely extract payload
    template<typename PayloadType>
    const PayloadType& get_payload() const {
        return std::get<PayloadType>(payload);
    }
    
    template<typename PayloadType>
    bool has_payload() const {
        return std::holds_alternative<PayloadType>(payload);
    }
};

} // namespace tpipeline
