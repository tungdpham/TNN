#pragma once

namespace tpipeline {

/**
 * @brief Enumeration of all possible command types in the pipeline system.
 * 
 * This enum defines the various types of messages that can be sent between
 * pipeline stages and the coordinator in both in-process and distributed scenarios.
 */
enum class CommandType {
    // Pipeline Tasks
    FORWARD_TASK,
    BACKWARD_TASK,
    
    // Control Commands
    START_TRAINING,
    STOP_TRAINING,
    PAUSE_TRAINING,
    RESUME_TRAINING,
    
    // Network Management
    HANDSHAKE_REQUEST,
    HANDSHAKE_RESPONSE,
    READY_SIGNAL,
    CONFIG_RECEIVED,
    
    // System State
    STATUS_REQUEST,
    STATUS_RESPONSE,
    HEALTH_CHECK,
    
    // Error Handling
    ERROR_REPORT,
    TASK_FAILURE,
    
    // Synchronization
    BARRIER_SYNC,
    CHECKPOINT_REQUEST,
    CHECKPOINT_COMPLETE,
    
    // Resource Management
    MEMORY_REPORT,
    RESOURCE_REQUEST,
    
    // Parameter Management
    UPDATE_PARAMETERS,
    PARAMETERS_UPDATED,
    
    // Stage Management
    QUERY_STAGE_INFO,
    STAGE_INFO_RESPONSE
};

} // namespace tpipeline
