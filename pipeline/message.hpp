#pragma once

#include "command_type.hpp"
#include "task.hpp"
#include <string>
#include <chrono>
#include <optional>

namespace tpipeline {

/**
 * @brief Simplified message structure focused on common use cases
 * 
 * Instead of a complex variant, we use separate optional fields
 * for different payload types. This makes the API much cleaner.
 */
template <typename T = float>
struct Message {
    CommandType command_type;
    
    // Core payload - most messages will use one of these
    std::optional<Task<T>> task;           // For FORWARD_TASK, BACKWARD_TASK
    std::optional<std::string> text_data;  // For simple string messages, errors, configs
    std::optional<bool> signal;            // For boolean signals (ready, success, etc.)
    
    // Routing information
    std::string sender_id;
    std::string recipient_id;
    
    // Metadata
    std::chrono::steady_clock::time_point timestamp;
    int sequence_number = 0;  // For ordering/deduplication
    
    // Constructors
    Message() {}

    Message(CommandType cmd_type)
        : command_type(cmd_type), timestamp(std::chrono::steady_clock::now()) {}
    
    // Task message constructor
    Message(CommandType cmd_type, const Task<T>& task_data)
        : command_type(cmd_type), task(task_data), 
          timestamp(std::chrono::steady_clock::now()) {}
    
    // Text message constructor  
    Message(CommandType cmd_type, const std::string& text)
        : command_type(cmd_type), text_data(text),
          timestamp(std::chrono::steady_clock::now()) {}
    
    // Signal message constructor
    Message(CommandType cmd_type, bool signal_value)
        : command_type(cmd_type), signal(signal_value),
          timestamp(std::chrono::steady_clock::now()) {}
    
    // Full constructor with routing
    template<typename PayloadType>
    Message(CommandType cmd_type, const PayloadType& payload, 
            const std::string& sender, const std::string& recipient)
        : command_type(cmd_type), sender_id(sender), recipient_id(recipient),
          timestamp(std::chrono::steady_clock::now()) {
        if constexpr (std::is_same_v<PayloadType, Task<T>>) {
            task = payload;
        } else if constexpr (std::is_same_v<PayloadType, std::string>) {
            text_data = payload;
        } else if constexpr (std::is_same_v<PayloadType, bool>) {
            signal = payload;
        }
    }

    // Helper methods
    bool is_task_message() const {
        return command_type == CommandType::FORWARD_TASK || command_type == CommandType::BACKWARD_TASK;
    }
    
    bool is_control_message() const {
        return command_type == CommandType::START_TRAINING ||
               command_type == CommandType::STOP_TRAINING ||
               command_type == CommandType::PAUSE_TRAINING ||
               command_type == CommandType::RESUME_TRAINING;
    }
    
    bool has_text() const { return text_data.has_value(); }

    bool has_signal() const { return signal.has_value(); }


    // Convenience factory methods
    static Message<T> forward_task(const Task<T>& task, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::FORWARD_TASK, task);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }
    
    static Message<T> backward_task(const Task<T>& task, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::BACKWARD_TASK, task);
        msg.sender_id = sender; 
        msg.recipient_id = recipient;
        return msg;
    }

    static Message<T> create_control_message(CommandType cmd_type, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(cmd_type);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }

    static Message<T> status_message(const std::string& status_text, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::STATUS_RESPONSE, status_text);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }

    static Message<T> parameters_updated(const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::PARAMETERS_UPDATED);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }
    
    static Message<T> ready_signal(const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::READY_SIGNAL, true);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }
    
    static Message<T> error_message(const std::string& error_text, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(CommandType::ERROR_REPORT, error_text);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }
    
    static Message<T> create_text_message(CommandType cmd_type, const std::string& text, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(cmd_type, text);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }
    
    static Message<T> create_signal_message(CommandType cmd_type, bool signal_value, const std::string& sender = "", const std::string& recipient = "") {
        Message<T> msg(cmd_type, signal_value);
        msg.sender_id = sender;
        msg.recipient_id = recipient;
        return msg;
    }

    std::string to_string() const {
        std::string result = "Message(" + std::to_string(static_cast<int>(command_type)) +
                             ", sender: " + sender_id + ", recipient: " + recipient_id;
        if (task) {
            result += ", task: " + task->to_string();
        }
        if (text_data) {
            result += ", text: " + *text_data;
        }
        if (signal) {
            result += ", signal: " + std::to_string(*signal);
        }
        result += ")";
        return result;
    }
};

} // namespace tpipeline