#!/usr/bin/env bash
set -euo pipefail

# ==== Helpers ====
require_env() {
  local var="$1"
  if [ -z "${!var+x}" ] || [ -z "${!var}" ]; then
    echo "ERROR: environment variable '$var' is required but not set (set it in docker-compose.yml)" >&2
    exit 2
  fi
}

detect_iface() {
  ip route | awk '/default/ {print $5; exit}'
}

# ==== Required env (ALL provided via docker-compose.yml) ====
require_env NETEM_ENABLE         # "1" or "0"
require_env NETEM_DELAY          # e.g. "0.5ms" (LAN tốt)
require_env NETEM_JITTER         # e.g. "0.1ms"
require_env NETEM_LOSS           # e.g. "0.01%"

require_env LOGGER_INTERVAL      # e.g. "0.01"
require_env LOGGER_DURATION      # e.g. "30"
require_env LOGGER_WARMUP        # e.g. "1"
require_env LOGGER_TAG           # e.g. "worker-8001"

# Optional: NETEM_RATE / NETEM_BURST / NETEM_LATENCY (nếu shaping băng thông)
# Optional: NETEM_IFACE (nếu không set sẽ tự dò)

# ==== Resolve interface ====
if [ -n "${NETEM_IFACE+x}" ] && [ -n "${NETEM_IFACE}" ]; then
  IFACE="$NETEM_IFACE"
else
  IFACE="$(detect_iface)"
  IFACE="${IFACE:-eth0}"  # fallback kỹ thuật
fi

# ==== Clean on exit ====
cleanup() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true
}
trap cleanup EXIT

# ==== Start CPU logger (background) ====
start_cpu_logger() {
  if [ -x /app/bin/cpu_logger ]; then
    mkdir -p /logs || true
    local outfile="/logs/${LOGGER_TAG}_$(date +%s).csv"
    echo "==> starting cpu_logger -> ${outfile}"
    /app/bin/cpu_logger \
      --interval "${LOGGER_INTERVAL}" \
      --duration "${LOGGER_DURATION}" \
      --outfile "${outfile}" \
      --tag "${LOGGER_TAG}" &
  else
    echo "[WARN] /app/bin/cpu_logger not found or not executable"
  fi
}

# ==== Apply netem ====
apply_netem() {
  tc qdisc del dev "$IFACE" root 2>/dev/null || true

  if [ "$NETEM_ENABLE" = "1" ]; then
    tc qdisc add dev "$IFACE" root handle 1: netem \
      delay "$NETEM_DELAY" "$NETEM_JITTER" \
      loss "$NETEM_LOSS" || true

    # Optional rate limit (only if ALL related vars are set)
    if [ -n "${NETEM_RATE+x}" ] && [ -n "${NETEM_RATE}" ]; then
      if [ -z "${NETEM_BURST+x}" ] || [ -z "${NETEM_BURST}" ] || \
         [ -z "${NETEM_LATENCY+x}" ] || [ -z "${NETEM_LATENCY}" ]; then
        echo "ERROR: NETEM_RATE set but NETEM_BURST/NETEM_LATENCY not provided." >&2
        exit 3
      fi
      tc qdisc add dev "$IFACE" parent 1: handle 10: tbf \
        rate "$NETEM_RATE" burst "$NETEM_BURST" latency "$NETEM_LATENCY" || true
    fi

    echo "==> tc qdisc on ${IFACE}:"
    tc qdisc show dev "$IFACE" || true
  else
    echo "==> NETEM disabled for ${IFACE}"
  fi
}

# ==== Exec flow ====
if [ "$#" -lt 1 ]; then
  echo "Usage: tcpem_cpu_logger.sh -- <command> [args...]" >&2
  exit 2
fi
[ "$1" = "--" ] && shift

start_cpu_logger
sleep "${LOGGER_WARMUP}"
apply_netem

echo "==> exec: $*"
exec "$@"
