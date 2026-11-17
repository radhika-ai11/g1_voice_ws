#!/usr/bin/env bash
# Benchmark the GITEX stall scene: how fast it runs (% of real time) and where a step's time goes.
#
#   bash training/bench_walk_scene.sh [SECONDS] [extra walk_scene.py args, e.g. --headless]
#
# Starts walk_scene.py --gitex --view stall with WALK_SCENE_PROFILE=1, lets it settle, then for SECONDS (default
# 120) walks the robot back and forth a little and runs one stall visit, the same every time. Prints the median
# speed and the median per-section breakdown. Stop any running scene first (only one can use the ROS topics).
set -eo pipefail
DURATION="${1:-120}"
shift || true
WARMUP=60
source /opt/ros/jazzy/setup.bash
LOG="$(mktemp -t bench_walk_scene.XXXX.log)"
cd "$HOME/g1_voice_ws/training"

WALK_SCENE_PROFILE=1 bash ../start_isaac.sh --gitex --view stall "$@" >"$LOG" 2>&1 &
SCENE=$!
trap 'kill $SCENE 2>/dev/null; wait $SCENE 2>/dev/null || true' EXIT

for _ in $(seq 600); do  # up to 10 minutes for Isaac Sim to start
  grep -q "\[walk_scene\] Walking" "$LOG" && break
  kill -0 $SCENE 2>/dev/null || { tail -30 "$LOG"; echo "scene exited"; exit 1; }
  sleep 1
done
echo "scene up; settling ${WARMUP}s" >&2
sleep "$WARMUP"
START_LINE=$(wc -l <"$LOG")

ros2 topic pub --once /gitex/visit std_msgs/msg/String "{data: apple}" >/dev/null
END=$((SECONDS + DURATION))
while [ $SECONDS -lt $END ]; do  # 5 s forward, 5 s back (slow), so the planner and IK do real work
  timeout 5 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}" >/dev/null 2>&1 || true
  timeout 5 ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.2}}" >/dev/null 2>&1 || true
done
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null

summarise() { python3 -c '
import re, statistics, sys
speeds, sections = [], {}
for line in sys.stdin:
    if m := re.search(r"simulation speed (\d+)%", line):
        speeds.append(int(m.group(1)))
    elif "per step:" in line:
        for name, ms in re.findall(r"([a-zA-Z ()]+?) ([\d.]+) ms", line.split("per step:")[1]):
            sections.setdefault(name.strip(" ,"), []).append(float(ms))
if not speeds:
    sys.exit("no speed reports")
print(f"median speed {statistics.median(speeds)}% of real time  (min {min(speeds)}%, max {max(speeds)}%, {len(speeds)} reports)")
for name, values in sorted(sections.items(), key=lambda item: -statistics.median(item[1])):
    print(f"  {name:32s} {statistics.median(values):6.1f} ms/step")
print("  (real-time budget: 20 ms/step; sections marked (in env step) are part of env step)")
' ; }
tail -n +"$((START_LINE + 1))" "$LOG" | summarise
echo "log: $LOG"
