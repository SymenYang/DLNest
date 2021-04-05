import time
import pynvml
try:
    from DeviceInformation import DeviceInformation
except ImportError:
    from .DeviceInformation import DeviceInformation

class GPUInformation(DeviceInformation):
    def __init__(self,ID):
        super(GPUInformation,self).__init__("GPU")
        self.ID = ID
        try:
            # 若卡信息获取失败，当作卡失效，设置isBreak = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.ID)
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.totalMemory = self.meminfo.total / 1024 / 1024
            self.isBreak = False
        except Exception as e:
            # 卡失效，totalMemory设为0
            self.totalMemory = 0
            self.isBreak = True

    def restartGPU(self):
        """
        重新尝试得到显卡信息，若成功，则返回True，同时修改isBreak，若失败则返回False
        """
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.ID)
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.totalMemory = self.meminfo.total / 1024 / 1024
            self.isBreak = False
            return True
        except Exception:
            self.isBreak = True
            return False

    def getFreeMemory(self):
        """
        return in MB
        """
        if self.isBreak:
            self.restartGPU()
        if self.isBreak:
            return 0
        try:
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return self.meminfo.free / 1024 / 1024
        except Exception as e:
            self.isBreak = True
            return 0

    def getDeviceStr(self):
        return "cuda:" + str(self.ID)