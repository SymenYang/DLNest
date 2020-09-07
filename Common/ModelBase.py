
class ModelBase():
    def __init__(self,args : dict,datasetInfo : dict = None):
        pass

    def initLog(self):
        return {}

    def getSaveDict(self):
        return {}

    def runOneStep(self,data,log : dict,iter : int,epoch : int):
        pass

    def visualize(self):
        pass

    def validate(self,valLoader):
        pass
