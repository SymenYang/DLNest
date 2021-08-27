import logging
from DLNest.Plugins.Tools import getArgs

class DLNestPlugin:
    def modelInit(self,args : dict, datasetInfo : dict = None):
        pluginName = "SimpleCMDVisualize"
        self._visualStride = getArgs(args, pluginName, "stride", 1)
        self._visualKeys = getArgs(args, pluginName, "keys" , [])
        self._visualFormat = getArgs(args, pluginName, "format", {})
        fmtDict = {key : "\t| {}: {}" for key in self._visualKeys}
        fmtDict.update(self._visualFormat)
        self._visualFormat = fmtDict
    
    def visualize(self, epoch : int, iter : int, log : dict):
        try:
            if iter % self._visualStride == 0:
                infoStr = ""
                for key in self._visualKeys:
                    if len(log[key]) > 0:
                        try:
                            infoStr = infoStr + self._visualFormat[key].format(key, log[key][-1])
                        except Exception as e:
                            logging.debug("[SimpleCMDVisualize]" + str(e))
                    else:
                        try:
                            infoStr = infoStr + self._visualFormat[key].format(key, None)
                        except Exception as e:
                            logging.debug("[SimpleCMDVisualize]" + str(e))
                logging.info("Iter : " + str(iter) + infoStr)
        except Exception as e:
            logging.debug("[SimpleCMDVisualize]" + str(e))