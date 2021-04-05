from DLNest.Executor.TrainProcess import TrainProcess
from DLNest.Information.TrainTask import TrainTask

def runIndependent(
    configPath : str,
    freqPath : str = "",
    devices : [int] = [-1],
    description : str = "",
    DDP : bool = False,
    noSave : bool = False,
    useDescriptionToSave : bool = False,
    showOnScreen : bool = True
):
    task = TrainTask.fromConfigFile(
        configPath = configPath,
        freqPath = freqPath,
        devices = devices,
        description = description,
        DDP = DDP,
        noSave = noSave,
        useDescriptionToSave = useDescriptionToSave
    )
    try:
        TProcess = TrainProcess(task,showOnScreen)
        TProcess.run()
    except KeyboardInterrupt:
        exit(0)