#!/bin/bash
#Fill username, ip adresses and number of nodes

ips=(0.0.0.0 0.0.0.0)
username=username
for x in {0..3}
do
  mv divided/femnist_$x.pickle divided/femnist.pickle
  scp divided/femnist.pickle $username@${ips[$x]}:fll/data/femnist
  mv divided/femnist.pickle divided/femnist_$x.pickle
done