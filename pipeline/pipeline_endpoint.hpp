#pragma once

#include <string>
#include <unordered_map>

namespace tpipeline {

/**
 * @brief Simple endpoint addressing information
 * 
 * Just contains the information needed to route messages,
 * not implementation details about how to communicate.
 */
struct StageEndpoint {
    std::string stage_id;           // Unique identifier like "stage_1", "coordinator"
    std::string communication_type; // "in_process", "tcp", "websocket", etc.
    std::string address;            // "localhost:8080", "memory", etc.
    std::unordered_map<std::string, std::string> parameters; // Extra config like ports, auth tokens
    
    StageEndpoint() = default;
    
    StageEndpoint(std::string id, std::string comm_type, std::string addr)
        : stage_id(std::move(id)), communication_type(std::move(comm_type)), 
          address(std::move(addr)) {}
    
    // Convenience constructor for in-process communication
    static StageEndpoint in_process(const std::string& id) {
        return StageEndpoint(id, "in_process", "memory");
    }
    
    // Convenience constructor for network communication
    static StageEndpoint network(const std::string& id, const std::string& host, int port) {
        StageEndpoint endpoint(id, "tcp", host + ":" + std::to_string(port));
        endpoint.parameters["host"] = host;
        endpoint.parameters["port"] = std::to_string(port);
        return endpoint;
    }
};

} // namespace tpipeline