# FLL - Federated Learning Library

Simple library created for a sole purpose of evaluating federated approach at training neural networks on distributed datasets. A unique feature that distinguishis it from other libraries designed to evaluate federated learing is possibility of real distribution among avaliable nodes, thanks to MPI.  

It is written in Python and is MPI based. To install necessary packages use:  
$ pip3 install -r requirements.txt

Also, MPI implementation is required e.g.
$ apt install libblacs-mpi-dev

Basic idea behind how FLL is creating a Process object in each process run. Those processes communicate using MPI. There is always one Server, and multiple Clients or MultiClients. 

There are 2 possible ways of running federated learning using this library, depending on the number of clients and on the size of datasets. 

1. If number of clients is small enough to symulate each client as a seperate process and dataset is small enough that 'server' process can load it and than distribute is among clients.
-Use Client class (not MulitClient),
-Provide full dataset, not divided into clients (downside of this approach is that dataset will be distributed randomly and equally among clients).

2. If number of clients in a dataset is very large or dataset is simply too big to be loaded by a single process:
-Use MultiClient class (not Client), each MultiClient process will simulate multiple clients,
-Divide the dataset (divided into client datasets) by yourselt and provide it to each node,
-Each client will have data divided by you, so you can distribute how you like),
-Usually, run one MultiClient process per node (though it is not required).

So far 3 example datasets and 3 coresponging scripts presenting what this library can do have been created:  
-gutenbergfll.py  
-mnistfll.py  
-femnist.py

<!---
## How it works

## gutenbergfll.py

## mnistfll.py
--->

FEMNIST:
Federated Extended MNIST is a dataset created by LEAF project. It consists of images (28x28x1) of hand written digits and letters. There are 62 classes in total (10 digits, 26 lowercase letters and 26 uppercase letters). Entire dataset is divided into 3500 clients, each client has a unique subset of these images, showing letters/numbers written by the same person. 

Example implementation of federated learning for femnist dataset is in femnist.py

It uses the second approach, where there are too many clients 
