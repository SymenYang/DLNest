import json
from pathlib import Path
import time

class Task:
    def __init__(
            self,
            modelFilePath : Path,
            datasetFilePath : Path,
            lifeCycleFilePath : Path,
            args : dict,
            otherFilePaths : list = [],
            timestamp : str = "",
            description : str = "",
            GPUID : int = 0,
            memoryConsumption : float = -1
        ):
        self.modelFilePath = modelFilePath
        self.datasetFilePath = datasetFilePath
        self.lifeCycleFilePath = lifeCycleFilePath
        self.otherFilePaths = otherFilePaths
        self.args = args
        self.timestamp = timestamp
        self.description = description
        self.GPUID = GPUID
        self.memoryConsumption = memoryConsumption

