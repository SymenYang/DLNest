
class DLNestPluginBase:
    _NANE = "DLNestPluginBase"
    _config = {}
    _defaultKeys = []

    @classmethod
    def getName(cls):
        return cls._NAME 
    
    @classmethod
    def getDefaultConfig(cls):
        ret = {}
        for key in cls._defaultKeys:
            ret[key] = cls._config[key]
        return ret
    
    @classmethod
    def getFullConfig(cls):
        return cls._config
    
    @classmethod
    def getArgs(cls, self, pluginName : str, name : str, default : any):
        args = self.getArgs()
        if "plugins_config" in args:
            pArgs = args["plugins_config"]
            if pluginName in pArgs:
                if name in pArgs[pluginName]:
                    return pArgs[pluginName][name]
                else:
                    return default
            else:
                return default
        else:
            return default