import time
import random
try:
    from ..SavePackage.SavePackage import SavePackage
except ImportError:
    from DLNest.SavePackage.SavePackage import SavePackage

class TaskInformation:
    def __init__(
        self,
        savePackage : SavePackage,
        devices : list = [],
        memoryConsumption : int = -1,
        multiGPU : bool = False,
        DDP : bool = False,
        loadCkpt : bool = False,
        checkpointID : int = -1
    ):
        self.ID = ("%.4f" % time.time())[6:] + "_" + str(random.randint(0,9))
        self.args = savePackage.args
        self.devices = devices
        self.memoryConsumption = memoryConsumption
        self.multiGPU = multiGPU
        self.DDP = DDP
        if self.DDP:
            self.multiGPU = True
        self.startTime = time.time()
        
        self.type = "None"
        self.status = "Pending"
        self.process = None

        self.port = "16484" # default port
        self.address = 'localhost' # default address
        self.savePackage = savePackage
        self.loadCkpt = loadCkpt
        self.checkpointID = checkpointID

        self.extraInfo = {}

    def getDict(self):
        return {
            "ID" : self.ID,
            "args" : self.args,
            "devices" : self.devices,
            "memory_consumption" : self.memoryConsumption,
            "multi_GPU" : self.multiGPU,
            "DDP" : self.DDP,
            "type" : self.type,
            "status" : self.status,
            "load_ckpt" : self.loadCkpt,
            "checkpoint_ID" : self.checkpointID
        }