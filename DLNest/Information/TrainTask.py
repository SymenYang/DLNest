try:
    from TaskInformation import TaskInformation
    from ..SavePackage.SavePackage import SavePackage
except ImportError:
    from DLNest.Information.TaskInformation import TaskInformation
    from DLNest.SavePackage.SavePackage import SavePackage

from multiprocessing import Queue

class TrainTask(TaskInformation):
    @classmethod
    def fromRecord(
        cls,
        recordPath : str,
        checkpointID : int = -1,
        devices : list = [],
        memoryConsumption : int = -1,
        multiGPU : bool = False,
        DDP : bool = False,
        description : str = ""
    ):
        # Get save package from existing package
        savePackage = SavePackage()
        savePackage.initFromAnExistSavePackage(recordPath)

        retTask = cls(
            savePackage = savePackage,
            devices = devices,
            memoryConsumption = memoryConsumption,
            multiGPU = multiGPU,
            DDP = DDP,
            description = description,
            loadCkpt = True,
            checkpointID = checkpointID
        )
        return retTask
    
    @classmethod
    def fromConfigFile(
        cls,
        configPath : str,
        freqPath : str = "",
        devices : list = [],
        memoryConsumption : int = -1,
        multiGPU : bool = False,
        DDP : bool = False,
        description : str = "",
        noSave : bool = False,
        useDescriptionToSave : bool = False
    ):
        # Get save package by config path
        savePackage = SavePackage(configPath = configPath,freqPath = freqPath)
        
        # Making special save dir name by noSave or useDescriptionToSave
        saveName = ""
        if noSave:
            saveName = "NOSAVE"
        elif useDescriptionToSave:
            saveName = description
        savePackage.saveToNewDir(saveName)

        retTask = cls(
            savePackage = savePackage,
            devices = devices,
            memoryConsumption = memoryConsumption,
            multiGPU = multiGPU,
            DDP = DDP,
            description = description,
            noSave = noSave,
            useDescriptionToSave = useDescriptionToSave,
            loadCkpt = False
        )
        return retTask

    def __init__(
        self,
        savePackage : SavePackage,
        devices : list = [],
        memoryConsumption : int = -1,
        multiGPU : bool = False,
        DDP : bool = False,
        description : str = "",
        noSave : bool = False,
        useDescriptionToSave : bool = False,
        loadCkpt : bool = False,
        checkpointID : int = -1
    ):
        super(TrainTask,self).__init__(
            savePackage = savePackage,
            devices = devices,
            memoryConsumption = memoryConsumption,
            multiGPU = multiGPU,
            DDP = DDP,
            loadCkpt = loadCkpt,
            checkpointID = checkpointID
        )
        self.ID = "T_" + self.ID
        self.description = description
        self.noSave = noSave
        self.useDescriptionToSave = useDescriptionToSave

        self.type = "Train"
        
        self.commandQueue = Queue()

        if description != "":
            savePackage.saveVisualString("desc: " + description)

    def getDict(self):
        ret = super().getDict()
        ret["description"] = self.description
        return ret

if __name__ == "__main__":
    TT = TrainTask.fromConfigFile("/root/code/DLNestTest/root_config.json")
    print(TT.getDict())
    TT = TrainTask.fromRecord("/root/code/DLNestTest/Saves/Some Description")
    print(TT.getDict())