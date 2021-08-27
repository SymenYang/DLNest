def getArgs(args : dict, pluginName : str, name : str, default : any):
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