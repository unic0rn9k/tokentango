#!/usr/bin/env fish
echo "Notifying $topic"

set pid (pgrep -f "train_mlm_comp")

while kill -0 $pid 2>/dev/null
    sleep 1
end

curl -d "finished trainig run 1 🎉" $topic

bash train_mlm_comp.sh 2>&1 | rg -v "\[=*>" | tee output.log

curl -T output.log $topic
