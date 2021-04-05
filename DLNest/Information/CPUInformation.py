try:
    from DeviceInformation import DeviceInformation
except ImportError:
    from .DeviceInformation import DeviceInformation
import psutil

class CPUInformation(DeviceInformation):
    def __init__(self):
        super(CPUInformation,self).__init__("CPU")
        nowMem = psutil.virtual_memory()
        self.totalMemory = nowMem.total / 1024 ** 2
    
    def getFreeMemory(self):
        nowMem = psutil.virtual_memory()
        return nowMem.available / 1024 ** 2

    def getDeviceStr(self):
        return "cpu"