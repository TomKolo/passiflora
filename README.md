# FLL - Federated Learning Library

### Simple library created for a sole purpose of evaluating federated approach at training neural networks on distributed datasets. A unique feature that distinguishis it from other libraries designed to evaluate federated learing is possibility of real distribution among avaliable nodes, thanks to MPI.  

#

## Instalation
Install required dependences

`pip install -r requirements.txt`

Install any MPI implementation, e.g.

`apt install libblacs-mpi-dev`

Install FLL library from .whl file

`pip install dist/fll-0.1.0-py3-none-any.whl`

or simply use fll from source files, they are located in examples/fll.

#

<!-- Basic idea behind how FLL is creating a Process object in each process run. Those processes communicate using MPI. There is always one Server, and multiple Clients or MultiClients. 

There are 2 possible ways of running federated learning using this library, depending on the number of clients and on the size of datasets. 

1. If number of clients is small enough to symulate each client as a seperate process and dataset is small enough that 'server' process can load it and than distribute is among clients.
-Use Client class (not MulitClient),
-Provide full dataset, not divided into clients (downside of this approach is that dataset will be distributed randomly and equally among clients).

2. If number of clients in a dataset is very large or dataset is simply too big to be loaded by a single process:
-Use MultiClient class (not Client), each MultiClient process will simulate multiple clients,
-Divide the dataset (divided into client datasets) by yourselt and provide it to each node,
-Each client will have data divided by you, so you can distribute how you like),
-Usually, run one MultiClient process per node (though it is not required). -->
