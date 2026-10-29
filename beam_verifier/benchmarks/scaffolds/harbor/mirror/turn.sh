# shellcheck shell=bash
# Arguments: instruction file, server log, address, startup/response timeouts,
# then the complete Mirror server command. Instructions remain file data.
set -euo pipefail
instruction_path=$1 server_log=$2 server_address=$3 ready_timeout=$4 response_timeout=$5
shift 5

if [[ ! -r "$instruction_path" ]]; then
  echo "No instruction at $instruction_path" >&2
  exit 1
fi
"$@" >"$server_log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" || true' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Use Bash's /dev/tcp support so the sandbox needs no separate HTTP client.
request() {
  local status
  exec 3<>"/dev/tcp/${server_address/:/\/}" || return 1
  cat >&3
  read -r -t "$1" status <&3 || {
    exec 3<&- 3>&-
    return 1
  }
  exec 3<&- 3>&-
  printf '%s' "$status"
}

ready=0
deadline=$((SECONDS + ready_timeout))
while ((SECONDS < deadline)); do
  if status=$(printf 'GET /health HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n' "$server_address" | request 1 2>/dev/null) &&
    [[ "$status" == *' 204'* ]]; then
    ready=1
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Mirror server exited before answering /health' >&2
    tail -n 20 "$server_log" >&2 || true
    exit 1
  fi
  sleep 0.1
done
if [[ "$ready" != 1 ]]; then
  echo "Mirror server never answered /health in ${ready_timeout}s" >&2
  tail -n 20 "$server_log" >&2 || true
  exit 1
fi

size=$(wc -c <"$instruction_path")
status=$(
  (
    printf 'POST /message HTTP/1.1\r\n'
    printf 'Host: %s\r\n' "$server_address"
    printf 'Content-Type: text/plain\r\n'
    printf 'Content-Length: %s\r\n' "$size"
    printf 'Connection: close\r\n\r\n'
    cat "$instruction_path"
  ) | request "$response_timeout"
) || status=''
case "$status" in
  *' 204'*) ;;
  *)
    echo "Mirror did not complete the turn: ${status:-no response}" >&2
    tail -n 20 "$server_log" >&2 || true
    exit 1
    ;;
esac
