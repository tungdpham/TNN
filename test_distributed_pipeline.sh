#!/bin/bash

# Script to test distributed pipeline

echo "=== Distributed Pipeline Test Script ==="

# Kill any existing network workers
echo "Cleaning up any existing processes..."
pkill -f network_worker
sleep 1

# Start network workers in background
echo "Starting network workers..."
./network_worker 8001 &
WORKER1_PID=$!
echo "Started worker on port 8001 (PID: $WORKER1_PID)"

./network_worker 8002 &
WORKER2_PID=$!
echo "Started worker on port 8002 (PID: $WORKER2_PID)"

./network_worker 8003 &
WORKER3_PID=$!
echo "Started worker on port 8003 (PID: $WORKER3_PID)"

./network_worker 8004 &
WORKER4_PID=$!
echo "Started worker on port 8004 (PID: $WORKER4_PID)"

# Wait a moment for workers to start
echo "Waiting 1 seconds for workers to initialize..."
sleep 1

# Check if workers are running
echo "Checking worker processes..."
ps aux | grep -E "network_worker (8001|8002|8003|8004)" | grep -v grep

# Run the distributed pipeline example
echo ""
echo "=== Running Distributed Pipeline Example ==="
./distributed_pipeline_example

# Cleanup
echo ""
echo "=== Cleaning up ==="
echo "Stopping network workers..."
kill $WORKER1_PID $WORKER2_PID $WORKER3_PID $WORKER4_PID 2>/dev/null
wait $WORKER1_PID $WORKER2_PID $WORKER3_PID $WORKER4_PID 2>/dev/null

echo "Test completed."
