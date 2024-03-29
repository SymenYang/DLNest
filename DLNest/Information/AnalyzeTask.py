try:
    from TaskInformation import TaskInformation
    from ..SavePackage.SavePackage import SavePackage
    from ..Output.AnalyzerBuffer import AnalyzerBuffer
except ImportError:
    from DLNest.Information.TaskInformation import TaskInformation
    from DLNest.SavePackage.SavePackage import SavePackage
    from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer
from pathlib import Path
from multiprocessing import Queue

class AnalyzeTask(TaskInformation):
    def __init__(
        self,
        recordPath : str,
        scriptPath : str = "",
        checkpointID : int = -1,
        devices : list = [],
        memoryConsumption : int = -1,
        DDP : bool = False
    ):
        savePackage = SavePackage()
        savePackage.initFromAnExistSavePackage(recordPath)
        super(AnalyzeTask,self).__init__(
            savePackage = savePackage,
            devices = devices,
            memoryConsumption = memoryConsumption,
            multiGPU = False,
            DDP = DDP,
            loadCkpt = True if checkpointID != -2 else False,
            checkpointID = checkpointID
        )
        self.ID = "A_" + self.ID
        self.recordPath = Path(recordPath)
        if scriptPath == "":
            self.scriptPath = self.recordPath.parent.parent / "AnalyzeScripts"
        else:
            self.scriptPath = Path(scriptPath)
        
        self.type = "Analyze"
        self.commandQueue = Queue()
        self.outputBuffer : AnalyzerBuffer = None
    
    def getDict(self):
        ret = super().getDict()
        ret["record_path"] = str(self.recordPath)
        ret["script_path"] = str(self.scriptPath)
        ret["checkpoint_ID"] = self.checkpointID
        return ret
