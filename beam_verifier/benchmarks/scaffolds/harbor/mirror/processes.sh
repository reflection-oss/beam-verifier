# shellcheck shell=bash
# Arguments: snapshot|stop, proc root, then baseline entries for stop.
action=$1 proc_root=$2
shift 2

# PID plus start time distinguishes reused PIDs. Zombies cannot modify the
# workspace and may never disappear under a PID 1 that does not reap them.
list_processes() {
  local stat line fields
  processes=()
  for stat in "$proc_root"/[0-9]*/stat; do
    IFS= read -r line 2>/dev/null <"$stat" || continue
    read -r -a fields <<<"${line##*) }"
    [[ "${fields[0]:-}" == Z ]] && continue
    stat=${stat%/stat}
    processes+=("${stat##*/} ${fields[19]:-}")
  done
}

case "$action" in
  snapshot)
    list_processes && printf '%s\n' "${processes[@]}"
    exit
    ;;
  stop) ;;
  *) exit 1 ;;
esac

# A missing or malformed baseline must never authorize stopping task processes.
if (($# == 0)); then
  echo "an empty baseline would make every process the turn's" >&2
  exit 1
fi
declare -A keep=([1]=1)
for entry in "$@"; do
  # A process name that breaks stat parsing leaves its start time empty.
  if [[ ! "$entry" =~ ^[0-9]+\ [0-9]*$ ]]; then
    printf 'not a baseline line: %q\n' "$entry" >&2
    exit 1
  fi
  keep[$entry]=1
done
ancestor=$$
while [[ -n "$ancestor" && "$ancestor" -gt 1 ]]; do
  keep[$ancestor]=1
  status=$ancestor ancestor=
  while IFS=$':\t ' read -r key value; do
    [[ "$key" == PPid ]] && {
      ancestor=$value
      break
    }
  done 2>/dev/null <"$proc_root/$status/status"
done
signal_turn() {
  local entry found=1
  list_processes
  for entry in "${processes[@]}"; do
    [[ -n "${keep[${entry%% *}]:-}" || -n "${keep[$entry]:-}" ]] && continue
    local signal
    for signal in "$@"; do
      kill "-$signal" "${entry%% *}" 2>/dev/null && found=0
    done
  done
  return "$found"
}

# Allow graceful exit, then repeatedly freeze and kill survivors and their forks.
signal_turn TERM CONT || exit 0
deadline=$((SECONDS + 5))
while ((SECONDS < deadline)); do
  sleep 0.1
  signal_turn 0 || exit 0
done
deadline=$((SECONDS + 30))
while ((SECONDS < deadline)); do
  signal_turn STOP KILL || exit 0
done
sleep 1
signal_turn 0 || exit 0
echo 'turn processes survived the stop' >&2
exit 1
