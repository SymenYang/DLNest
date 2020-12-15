from .Core.DLNestCore import DLNestCore
from .Core.TrainProcess import TrainProcess
from .BridgeLayers.InformationLayer import TrainTask
import json
from pathlib import Path
import argparse
import time

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class TaskArguments(Arguments):
    def __init__(self):
        super(TaskArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.",required=True)
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-ns",action='store_true', help="Use to save to the NOSAVE dir")
        self._parser.add_argument("-ss",action='store_true', help="Use to also show on screen")

def runTrain(
        rootConfig : str, 
        description : str = "",
        freqConfig : str = "", 
        noSave : bool = False,
        showOnScreen : bool = False
    ):
    Core = DLNestCore()
    taskArgsPath = Path(rootConfig)
    taskArgs = {}
    Core._DLNestCore__loadArgs(taskArgsPath,taskArgs)
    if freqConfig != "":
        hotArgsPath = Path(freqConfig)
        Core._DLNestCore__loadArgs(hotArgsPath,taskArgs)
        
    task = TrainTask(
        args = taskArgs,
        description=description
    )
    nowTime = time.time()
    if noSave:
        task.timestamp = "NOSAVE"
    else:
        task.timestamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(nowTime)) + "_0"
    
    taskFakeProcess = TrainProcess(task,showOnScreen=showOnScreen)
    taskFakeProcess.run()

if __name__ == "__main__":
    argparser = TaskArguments()
    parser = argparser.parser()
    args = parser.parse_args()
    runTrain(
        args.c,
        args.d,
        args.f,
        args.ns,
        args.ss
    )
