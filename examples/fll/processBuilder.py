from . import Process, DEBUG
from . import Server
from . import Client
from . import MultiClient
from mpi4py import MPI

class ProcessBuilder:
    @staticmethod
    def build_process(delay_function, multi_client=False):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        delay = delay_function()
        device_name = MPI.Get_processor_name()
        if size == 1:
            raise Exception("No clients can be created, there is only one process!")
        
        if rank == 0:
            if DEBUG:
                print("Creating server of rank " + str(rank) + " in process pool " + str(size) + "processor name (id):" + MPI.Get_processor_name())
            return Server(rank, size, comm, delay, device_name, multi_client=multi_client)
        else:
            if multi_client == True:
                if DEBUG:
                    print("Creating client of rank " + str(rank) + " in process pool " + str(size) + "processor name (id):" + MPI.Get_processor_name())
                return MultiClient(rank, comm, delay, device_name)
            else:
                if DEBUG:
                    print("Creating client of rank " + str(rank) + " in process pool " + str(size) + "processor name (id):" + MPI.Get_processor_name())
                return Client(rank, comm, delay, device_name)