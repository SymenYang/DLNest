from DLNest.Common.RunnerBase import RunnerBase

class Runner(RunnerBase):
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

    def validationInit(self):
        pass

    def validateABatch(self,data, iter : int):
        pass

    def validationAnalyze(self, log : dict):
        pass