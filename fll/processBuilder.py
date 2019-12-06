from . import Process
from . import Server
from . import Client
from mpi4py import MPI
DEBUG = True
class ProcessBuilder:
    @staticmethod
    def buildProcess():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size == 1:
            raise Exception("No clients can be created, there is only one process!")
        
        if rank == 0:
            if DEBUG:
                print("Creating server of rank " + str(rank) + " in process pool " + str(size))
            return Server(rank, size, comm)
        else:
            if DEBUG:
                print("Creating client of rank " + str(rank) + " in process pool " + str(size))
            return Client(rank, comm)