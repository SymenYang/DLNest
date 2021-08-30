from functools import wraps
import logging

class ModelBase:
    def __init__(self,_envType : str, args : dict,rank = -1,worldSize = -1, plugins : list = []):
        self._envType = _envType
        self._rank = rank
        self._worldSize = worldSize
        self._plugins = plugins
        self._args = args

    def getArgs(self):
        return self._args

    def checkPlugins(func):
        @wraps(func)
        def checkAndRun(*args, **kwargs):
            name = func.__name__
            for plugin in args[0]._plugins:
                if name[1:] in dir(plugin):
                    try:
                        getattr(plugin,name[1:])(*args, **kwargs)
                    except Exception as e:
                        logging.debug(str(e))
            return func(*args, **kwargs)
        
        return checkAndRun

    def checkDictOutputPlugins(func):
        @wraps(func)
        def checkAndRun(*args, **kwargs):
            name = func.__name__
            ret = {}
            for plugin in args[0]._plugins:
                if name[1:] in dir(plugin):
                    try:
                        ret.update(getattr(plugin,name[1:])(*args, **kwargs))
                    except Exception as e:
                        logging.debug(str(e))
            ret.update(func(*args, **kwargs))
            return ret
        
        return checkAndRun

    @checkPlugins
    def _modelInit(self,args : dict, datasetInfo : dict = None):
        return self.init(args, datasetInfo)

    def init(self,args : dict, datasetInfo : dict = None):
        self.args = args
        pass

    @checkPlugins
    def _initOptimizer(self):
        return self.initOptimizer()

    def initOptimizer(self):
        pass
    
    def DDPOperation(self,rank : int):
        pass

    def afterDDP(self, rank : int):
        """
        For SyncBN or something else.
        """
        pass

    @checkDictOutputPlugins
    def _initLog(self):
        return self.initLog()

    def initLog(self):
        return {}

    @checkDictOutputPlugins
    def _getSaveDict(self):
        return self.getSaveDict()
    
    def getSaveDict(self):
        return {}

    @ checkPlugins
    def _loadSaveDict(self,saveDict):
        return self._loadSaveDict(saveDict = saveDict)

    def loadSaveDict(self,saveDict):
        pass

    @checkPlugins
    def _runOneStep(self,data,log : dict, iter : int, epoch : int):
        return self.runOneStep(data = data, log = log, iter = iter, epoch = epoch)

    def runOneStep(self,data,log : dict, iter : int, epoch : int):
        pass

    @checkPlugins
    def _visualize(self,log : dict, iter : int, epoch : int):
        return self.visualize(log = log, iter = iter, epoch = epoch)

    def visualize(self,log : dict, iter : int, epoch : int):
        pass

    @checkPlugins
    def _validationInit(self):
        return self.validationInit()

    def validationInit(self):
        pass

    @checkPlugins
    def _validateABatch(self,data, iter : int):
        return self.validateABatch(data = data, iter = iter)

    def validateABatch(self,data, iter : int):
        pass

    @checkPlugins
    def _validationAnalyze(self, log : dict):
        return self.validationAnalyze(log = log)

    def validationAnalyze(self, log : dict):
        pass


class test_plugin:
    def validationInit(self):
        print(self)

    def visualize(self, log: dict, iter: int, epoch: int):
        print(self._envType,log,iter,epoch)
    
    def initLog(self):
        return {
            "test" : []
        }

class test_plugin2:
    def initLog(self):
        return {
            "test2" : []
        }

if __name__ == "__main__":
    M = ModelBase("haha",plugins=[test_plugin,test_plugin2])
    M._visualize({'test' : "test"},123,321)
    M._validationInit()
    print(M._initLog())