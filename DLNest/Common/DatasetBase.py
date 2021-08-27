try:
    import torch
except ImportError:
    pass
from functools import wraps

class DatasetBase:
    def __init__(self,_envType : str, plugins : list = []):
        self._envType = _envType
        self._plugins = plugins

    def checkPlugins(func):
        @wraps(func)
        def checkAndRun(*args, **kwargs):
            name = func.__name__
            for plugin in args[0]._plugins:
                if name[1:] in dir(plugin):
                    getattr(plugin,name[1:])(*args, **kwargs)
            return func(*args, **kwargs)
        
        return checkAndRun

    def checkDictOutputPlugins(func):
        @wraps(func)
        def checkAndRun(*args, **kwargs):
            name = func.__name__
            ret = {}
            for plugin in args[0]._plugins:
                if name[1:] in dir(plugin):
                    ret.update(getattr(plugin,name[1:])(*args, **kwargs))
            ret.update(func(*args, **kwargs))
            return ret
        
        return checkAndRun

    @checkPlugins
    def _datasetInit(self, args : dict):
        return self.init(args = args)

    def init(self,args : dict):
        """
        input:
            args : dict
        output:
            dict to model
            train loader
            val loader
        """
        return {},None,None

    def getSampler(self, dataset):
        if self._envType == "DDP":
            return torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            return None

    @checkDictOutputPlugins
    def _getSaveDict(self):
        return self.getSaveDict()

    def getSaveDict(self):
        return {}
    
    @checkPlugins
    def _loadSaveDict(self, saveDict):
        return self.loadSaveDict(saveDict = saveDict)

    def loadSaveDict(self,saveDict):
        pass