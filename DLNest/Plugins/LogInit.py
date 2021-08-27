import logging
from DLNest.Plugins.Tools import getArgs

class DLNestPlugin:
    def BAll(self):
        import sys
        log = logging.getLogger()
        while len(log.handlers) > 0:
            log.removeHandler(log.handlers[0])
        
        args = self.taskProcess.task.args
        level = getArgs(args,"LogInit","level", "INFO")
        format = getArgs(args,"LogInit","format", "[%(asctime)s][%(levelname)s] %(message)s")
        datefmt= getArgs(args,"LogInit","datefmt", "%Y-%m-%d %H:%M:%S")
        logging.basicConfig(level = level, format = format,datefmt = datefmt)