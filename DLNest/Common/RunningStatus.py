from enum import Enum

DLNestStatus = Enum("DLNestStatus", ("Training","Validating","Analyzing","Waiting"))

class RunningStatus:
    def __init__(self):
        self.__epoch = 0
        self.__iter = 0
        self.__status = DLNestStatus.Waiting
        self.__env = "CPU"
        self.__rank = -1
        self.__worldSize = 0
    
    @property
    def epoch(self):
        return self.__epoch
    
    @property
    def iter(self):
        return self.__iter
    
    @property
    def status(self):
        return self.__status
    
    @property
    def env(self):
        return self.__env
    
    @property
    def rank(self):
        return self.__rank
    
    @property
    def worldSize(self):
        return self.__worldSize

    def isTraining(self):
        return self.status == DLNestStatus.Training
    
    def isValidating(self):
        return self.status == DLNestStatus.Validating
    
    def isAnalyzing(self):
        return self.status == DLNestStatus.Analyzing
    
    def isWaiting(self):
        return self.status == DLNestStatus.Waiting

    def startTraining(self):
        self.__status = DLNestStatus.Training

    def startValidating(self):
        self.__status = DLNestStatus.Validating
    
    def startAnalyzing(self):
        self.__status = DLNestStatus.Analyzing
    
    def startWaiting(self):
        self.__status = DLNestStatus.Waiting