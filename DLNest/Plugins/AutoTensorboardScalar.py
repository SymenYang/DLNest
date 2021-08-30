from torch.utils.tensorboard import SummaryWriter
import logging
from DLNest.Plugins.DLNestPluginBase import DLNestPluginBase as DPB

class DLNestPlugin(DPB):
    _NAME = "AutoTensorboardScalar"
    _config = {}
    _defaultKeys = []
    def runnerInit(self,args : dict, datasetInfo : dict = None):
        if self._rank == -1 or self._rank == 0:
            self.writer = SummaryWriter(".")
        logging.debug("[AutoTensorboardScalar] Finish modelInit")
    
    def visualize(self, log : dict, epoch : int, iter : int):
        if self._rank == -1 or self._rank == 0:
            if not "_lastLen" in dir(self):
                self._lastLen = {key : 0 for key in log}

            for key in log:
                if isinstance(log[key], list) and len(log[key]) != self._lastLen[key]:
                    try:
                        self.writer.add_scalar(key,log[key][-1],len(log[key]) - 1)
                        self._lastLen[key] = len(log[key])
                    except Exception as e:
                        logging.debug("[AutoTensorboardScalar]" + str(e))
            logging.debug("[AutoTensorboardScalar] Finish visualize")