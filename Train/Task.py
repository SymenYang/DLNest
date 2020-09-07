import json
from pathlib import Path
import time

class Task:
    def __init__(
            self,
            modelFilePath : Path,
            datasetFilePath : Path,
            lifeCycleFilePath : Path,
            otherFilePath : list,
            args : dict,
            timestamp : str,
            description : str
        ):
        self.modelFilePath = modelFilePath
        self.datasetFilePath = datasetFilePath
        self.lifeCycleFilePath = lifeCycleFilePath
        self.otherFilePath = otherFilePath
        self.args = args
        self.timestamp = timestamp
        self.description = description

