from DLNest.Common.ModelBaseTorch import ModelBase

class Model(ModelBase):
    def init(self,args : dict,datasetInfo : dict = None):
        #Init models
        pass

    def initLog(self):
        return {}

    def initOptimizer(self):
        # init optimizers
        pass

    def runOneStep(self,data, log : dict, iter : int, epoch : int):
        pass

    def visualize(self,epoch : int, iter : int, log : dict):
        pass

    def validate(self,valLoader,log : dict):
        pass