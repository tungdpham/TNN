PROFILE=single-model

# parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --profile | -p)
      PROFILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--profile <profile_name>]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -n "$PROFILE" ]]; then
  echo "Starting detached docker containers with profile: $PROFILE"
  docker-compose --profile "$PROFILE" up -d
else
  echo "Need to specify a profile"
  exit 1
fi