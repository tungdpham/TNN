#!/bin/sh

# Check the PROFILE environment variable and run the corresponding program
case "$PROFILE" in
  sync)
    exec ./sync_pipeline_coordinator
    ;;
  semi_async)
    exec ./semi_async_pipeline_coordinator
    ;;
  async)
    echo "Async mode is not yet supported."
    exit 1
    ;;
  *)
    echo "Invalid PROFILE: $PROFILE. Supported values are 'sync', 'semi_async'."
    exit 1
    ;;
esac