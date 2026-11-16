#!/usr/bin/env bash
# ops/vm_go.sh — run on the laptop:  bash code/ops/vm_go.sh <window>
# Resets the VM checkout to origin/main, starts vm_run.sh in tmux there (which pushes and
# shuts the VM down when done), and tails the run log.
set -u
W="${1:-base_covid}"
S="gcloud compute ssh chariag1_ramzi@instance-20260808-143737 --zone=us-central1-c --project=roysearch"

$S --command="
cd ~/DMP_labour_market_choice
git fetch -q origin && git checkout -q main && git reset -q --hard origin/main
tmux kill-session -t roysearch 2>/dev/null
tmux new -d -s roysearch \"cd ~/DMP_labour_market_choice && ROYSEARCH_WINDOW=${W} bash code/ops/vm_run.sh\"
sleep 5; tmux ls
"
sleep 3
$S --command="tail -n 999999 -f ~/DMP_labour_market_choice/output/logs/${W}_vm.log"
