try:
    import torch
except ImportError:
    pass
from functools import wraps
import logging

class DatasetBase:
    def __init__(self,_envType : str, args : dict, plugins : list = []):
        self._envType = _envType
        self._args = args
        self._plugins = plugins

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