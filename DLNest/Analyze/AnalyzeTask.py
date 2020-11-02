import json
from pathlib import Path

class AnalyzeTask:
    def __init__(
            self,
            recordPath : Path,
            scriptPath : Path,
            checkpointID : int,
            GPUID : int = 0,
            memoryConsumption : float = -1
        ):
        self.recordPath = recordPath
        self.scriptPath = scriptPath
        self.GPUID = GPUID
        self.checkpointID = checkpointID
        self.memoryConsumption = memoryConsumption

