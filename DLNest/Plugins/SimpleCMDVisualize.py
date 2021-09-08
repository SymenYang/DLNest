import logging
from DLNest.Plugins.DLNestPluginBase import DLNestPluginBase as DPB

class DLNestPlugin(DPB):
    _NAME = "SimpleCMDVisualize"
    _config = {
        "stride" : 1,
        "keys" : [],
        "format" : {}
    }
    _defaultKeys = ["stride", "keys", "format"]

    def runnerInit(self,args : dict, datasetInfo : dict = None):
        pluginName = "SimpleCMDVisualize"
        self._visualStride = DPB.getArgs(self, pluginName, "stride", 1)
        self._visualKeys = DPB.getArgs(self, pluginName, "keys" , [])
        self._visualFormat = DPB.getArgs(self, pluginName, "format", {})
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