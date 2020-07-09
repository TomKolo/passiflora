#!/bin/bash
#Fill username nad ips adresses

ips=(0.0.0.0 0.0.0.0)
username=username
for x in {1..${#ips[@]}}
do
  mv divided/femnist_$x.pickle divided/femnist.pickle
  scp divided/femnist.pickle $username@:${ips[$i]}/fll/data/femnist
  mv divided/femnist.pickle divided/femnist$x.pickle
done