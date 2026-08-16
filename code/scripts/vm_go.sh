#!/usr/bin/env bash
# ON THE LAPTOP, from the repo:  bash code/scripts/vm_go.sh base_covid
# Accesses the VM, starts the run (auto push + auto shutdown), tails the log.
set -u
W="${1:-base_covid}"
S="gcloud compute ssh chariag1_ramzi@instance-20260808-143737 --zone=us-central1-c --project=roysearch"

$S --command="
cd ~/DMP_labour_market_choice
git fetch -q origin && git checkout -q main && git reset -q --hard origin/main
tmux kill-session -t roysearch 2>/dev/null
tmux new -d -s roysearch \"cd ~/DMP_labour_market_choice && ROYSEARCH_WINDOW=${W} bash code/scripts/vm_run.sh\"
sleep 5; tmux ls
"
sleep 3
$S --command="tail -n 999999 -f ~/DMP_labour_market_choice/output/logs/${W}_vm.log"
