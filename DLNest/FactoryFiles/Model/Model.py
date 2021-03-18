from DLNest.Common.ModelBase import ModelBase

class Model(ModelBase):
    def __init__(self,args : dict,datasetInfo : dict = None):
        pass

    def DDPOperation(self,rank : int,world_size : int):
        pass

    def initLog(self):
        return {}

    def getSaveDict(self):
        return {}

    def loadSaveDict(self,saveDict : dict):
        pass

    def runOneStep(self,data,log : dict,iter : int,epoch : int):
        pass

    def visualize(self, log : dict, iter : int, epoch : int):
        pass

    def validate(self,valLoader,log : dict):
        pass
