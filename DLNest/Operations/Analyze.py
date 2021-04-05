from DLNest.Scheduler.Scheduler import Scheduler
from DLNest.Information.AnalyzeTask import AnalyzeTask

def analyze(
    recordPath : str,
    scriptPath : str = "",
    checkpointID : int = -1,
    CPU : bool = False,
    memoryConsumption : int = -1,
    otherArgs : dict = {}
):
    devices = []
    if CPU:
        devices = [-1]
    task = AnalyzeTask(
        recordPath = recordPath,
        scriptPath = scriptPath,
        checkpointID = checkpointID,
        memoryConsumption = memoryConsumption,
        devices = devices
    )

    scheduler = Scheduler()
    scheduler.giveATask(task,otherArgs = otherArgs)