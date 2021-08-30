import logging
from DLNest.Plugins.DLNestPluginBase import DLNestPluginBase as DPB

class DLNestPlugin(DPB):
    _NAME = "LogInit"
    _config = {
        "level" : "INFO",
        "format" : "[%(asctime)s][%(levelname)s] %(message)s",
        "datefmt" : "%Y-%m-%d %H:%M:%S"
    }
    _defaultKeys = ["level"]
    def BAll(self):
        import sys
        log = logging.getLogger()
        while len(log.handlers) > 0:
            log.removeHandler(log.handlers[0])
        
        args = self.taskProcess.task.args
        pluginName = "LogInit"
        level = DPB.getArgs(self, pluginName, "level", "INFO")
        format = DPB.getArgs(self, pluginName, "format", "[%(asctime)s][%(levelname)s] %(message)s")
        datefmt= DPB.getArgs(self, pluginName, "datefmt", "%Y-%m-%d %H:%M:%S")
        logging.basicConfig(level = level, format = format,datefmt = datefmt)