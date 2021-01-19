# FEMNIST DATASET

## Femnist (Federated Extended MNIST) is a dataset created by LEAF (https://github.com/TalwalkarLab/leaf). It consists of images of hand written digits and letters divided into 3550 client subsets. Each subset contains signs written by one person.

#

## Script downloading raw data.
`./download_data.sh`

Script downloading raw FEMNIST dataset. A total of 35 files is downloaded, each containing about 100 client datasets.

## Script spliting raw data
`python split_femnist.sh -n 20 -p 80`

Script spliting raw dataset into given number of files.

-n number of files

-p percentage of original dataset to be used


## Script distributing files with client datasets among nodes
### Note: Uses SSH protocol

`./distribute_femnist.sh`

Script distributing files with client datasets among nodes. In order to use define number_of_nodes and their ip adresses in the script.