#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: run_acceptance.sh PIBT_LIBRARY LACAM_WORKER HOSPITAL_SOURCES NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
export WM_PIBT_LIBRARY="$(realpath "$1")"
export WM_LACAM3_WORKER="$(realpath "$2")"
export WM_HOSPITAL_SOURCES="$(realpath "$3")"
WM_DEMO_OUT="$4"
if [[ -e "$WM_DEMO_OUT" ]] && [[ -n "$(ls -A "$WM_DEMO_OUT")" ]]; then
  echo "refusing to overwrite $WM_DEMO_OUT" >&2
  exit 2
fi
mkdir -p "$WM_DEMO_OUT"
timeout 600s python -m business_transport.demo compare --output "$WM_DEMO_OUT/comparison"
timeout 180s python -m business_transport.demo run --scenario clinic --seed 61000 \
  --policy nearest --motion hybrid --output "$WM_DEMO_OUT/clinic"
timeout 180s python -m business_transport.demo run --scenario three_stage --motion legacy \
  --policy workflow_rule --output "$WM_DEMO_OUT/three-stage"
timeout 180s python -m business_transport.demo run --scenario empty8 --regime workflow_outage --seed 71003 \
  --policy nearest --motion hybrid --output "$WM_DEMO_OUT/outage"
python -m experiments.system_demo.verify "$WM_DEMO_OUT"
for case_path in comparison/nearest comparison/workflow_rule comparison/workflow_pets comparison/workflow_structured clinic three-stage outage; do
  node tests/test_business_playback.js "$WM_DEMO_OUT/$case_path"
done
