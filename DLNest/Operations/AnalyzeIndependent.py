from DLNest.Executor.AnalyzeProcess import AnalyzeProcess
from DLNest.Information.AnalyzeTask import AnalyzeTask

def AnalyzeIndependent(
    recordPath : str,
    expFunc = None,
    scriptPath : str = "",
    checkpointID : int = -1,
    devices : list = []
):
    task = AnalyzeTask(
        recordPath = recordPath,
        scriptPath = scriptPath,
        checkpointID = checkpointID,
        devices = devices
    )
    try:
        AProcess = AnalyzeProcess(task,expFunc=expFunc)
        AProcess.run()
    except KeyboardInterrupt:
        exit(0)

