#!/bin/bash

# Script to profile network workers and pipeline coordinator using samply
# Usage: ./profile_with_samply.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting profiling session with samply...${NC}"

# Create profiles directory if it doesn't exist
mkdir -p profiles

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Cleaning up background processes...${NC}"
    jobs -p | xargs -r kill
    wait
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

echo -e "${YELLOW}Starting network workers and coordinator...${NC}"

# Start network_worker on port 8001 with samply in background
echo -e "${GREEN}Starting network_worker on port 8001${NC}"
samply record -o profiles/network_worker_8001_$(date +%Y%m%d_%H%M%S).json ./bin/network_worker 8001 &
WORKER1_PID=$!

# Start network_worker on port 8002 with samply in background  
echo -e "${GREEN}Starting network_worker on port 8002${NC}"
samply record -o profiles/network_worker_8002_$(date +%Y%m%d_%H%M%S).json ./bin/network_worker 8002 &
WORKER2_PID=$!

sleep 2

# Start semi_async_pipeline_coordinator with samply
echo -e "${GREEN}Starting semi_async_pipeline_coordinator${NC}"
samply record -o profiles/semi_async_coordinator_$(date +%Y%m%d_%H%M%S).json ./bin/semi_async_pipeline_coordinator &
COORDINATOR_PID=$!

echo -e "${GREEN}All processes started successfully!${NC}"
echo -e "${YELLOW}Process IDs:${NC}"
echo -e "  Network Worker 8001: $WORKER1_PID"
echo -e "  Network Worker 8002: $WORKER2_PID" 
echo -e "  Pipeline Coordinator: $COORDINATOR_PID"

echo -e "\n${YELLOW}Press Ctrl+C to stop all processes and save profiles...${NC}"

# Wait for all background jobs to complete or be interrupted
wait

echo -e "${GREEN}Profiling session completed!${NC}"
echo -e "${YELLOW}Profile files saved in ./profiles/ directory${NC}"
ls -la profiles/*.json 2>/dev/null || echo "No profile files found"
