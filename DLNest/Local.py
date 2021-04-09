from DLNest.Operations.Analyze import analyze
from DLNest.Operations.ChangeDelay import changeDelay
from DLNest.Operations.ChangeDevices import changeDevices
from DLNest.Operations.ChangeMaxTaskPerDevice import changeMaxTaskPerDevice
from DLNest.Operations.ContinueTrain import continueTrain
from DLNest.Operations.DelATask import delATask
from DLNest.Operations.GetAnalyzeOutput import getAnalyzeOutput
from DLNest.Operations.GetDevicesInformation import getDevicesInformation
from DLNest.Operations.GetTasksInformation import getTasksInformation
from DLNest.Operations.New import new
from DLNest.Operations.Run import run
from DLNest.Operations.RunExp import runExp
from DLNest.Operations.SafeExit import safeExit

import argparse
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import traceback

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class TrainArguments(Arguments):
    def __init__(self):
        super(TrainArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.",required = True)
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-ns",action='store_true', help="Set to save to the NOSAVE dir.")
        self._parser.add_argument("-mc",action='store_true',help="Set to use multi card.")
        self._parser.add_argument("-sd",action='store_true',help="Set to use description as the save dir name.(coverd by ns)")
        self._parser.add_argument("-DDP",action='store_true',help="Set to use DDP.")
        self._parser.add_argument("-CPU",action='store_true',help="Set to use CPU.")

class ProjectArguments(Arguments):
    def __init__(self):
        super(ProjectArguments, self).__init__(desc="Arguments for create a DLNest project.")

        self._parser.add_argument("-d",type=str, help="Path to the directory you want to create the project.", required = True)
        self._parser.add_argument("-MNIST",action='store_true', help="Set to new a project with MNIST task.")

class AnalyzeArguments(Arguments):
    def __init__(self):
        super(AnalyzeArguments, self).__init__(desc="Arguments for an Analyzer")

        self._parser.add_argument("-r",type=str, help = "path to the model record directory.", required = True)
        self._parser.add_argument("-s",type=str, default = "", help = "path to the analyze scripts.")
        self._parser.add_argument("-c",type=int, default = -1, help = "which epoch you want the model to load.(int)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-CPU",action='store_true',help="Set to use CPU.")

class ContinueArguments(Arguments):
    def __init__(self):
        super(ContinueArguments, self).__init__(desc="Arguments for an Analyzer")

        self._parser.add_argument("-r",type=str, help = "path to the model record directory.", required = True)
        self._parser.add_argument("-c",type=int, default = -1, help = "which epoch you want the model to load.(int)")
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-CPU",action='store_true',help="Set to use CPU.")
        self._parser.add_argument("-DDP",action='store_true',help="Set to use DDP.")
        self._parser.add_argument("-mc",action='store_true',help="Set to use multi card.")

class DeviceChangeArguments(Arguments):
    def __init__(self):
        super(DeviceChangeArguments, self).__init__(desc="Arguments for change valid cards.")
        self._parser.add_argument("-d",type=int, nargs='+', help='valid devices', required = True)

class DLNestLocal:
    def __init__(self):
        self.trainArgParser = TrainArguments()
        self.continueArgParser = ContinueArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()
        self.deviceChangeArgParser = DeviceChangeArguments()

    def runTrain(self,commandWordList : list):
        args,otherArgs = self.trainArgParser.parser().parse_known_args(commandWordList[1:])
        run(
            configPath = args.c,
            freqPath = args.f,
            description = args.d,
            memoryConsumption = args.m,
            CPU = args.CPU,
            DDP = args.DDP,
            multiGPU = args.mc,
            noSave = args.ns,
            useDescriptionToSave = args.sd,
            otherArgs = {}
        )

    def newProject(self,commandWordList : list):
        args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
        new(
            targetDir = args.d,
            MNIST = args.MNIST
        )

    def runAnalyze(self,commandWordList : list):
        args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
        analyze(
            recordPath = args.r,
            scriptPath = args.s,
            checkpointID = args.c,
            CPU = args.CPU,
            memoryConsumption = args.m,
            otherArgs = {}
        )

    def continueTrain(self,commandWordList : list):
        args,otherArgs = self.continueArgParser.parser().parse_known_args(commandWordList[1:])
        continueTrain(
            recordPath = args.r,
            checkpointID = args.c,
            memoryConsumption = args.m,
            CPU = args.CPU,
            DDP = args.DDP,
            multiGPU = args.mc,
            description = args.d,
            otherArgs = {}
        )

    def changeDevices(self,commandWordList : list):
        args,otherArgs = self.deviceChangeArgParser.parser().parse_known_args(commandWordList[1:])
        changeDevices(args.d)

    def runExp(self,commandWordList : list):
        runExp(commandWordList[1],commandWordList[2])

    def run(self):
        self.session = PromptSession(auto_suggest = AutoSuggestFromHistory())
        while True:
            try:
                command = self.session.prompt(HTML("<seagreen><b>DLNest>></b></seagreen>"))
                commandWordList = command.strip().split(' ')
                if commandWordList[0] == "run":
                    self.runTrain(commandWordList)
                elif commandWordList[0] == "continue":
                    self.continueTrain(commandWordList)
                elif commandWordList[0] == "new":
                    self.newProject(commandWordList)
                elif commandWordList[0] == "analyze":
                    self.runAnalyze(commandWordList)
                elif commandWordList[0] == "runExp":
                    self.runExp(commandWordList)
                elif commandWordList[0] == "del":
                    delATask(commandWordList[1])
                elif commandWordList[0] == "showAN":
                    print(getAnalyzeOutput(commandWordList[1],False)[1])
                elif commandWordList[0] == "showTask":
                    print(getTasksInformation())
                elif commandWordList[0] == "showDevice":
                    print(getDevicesInformation())
                elif commandWordList[0] == 'changeDevices':
                    self.changeDevices(commandWordList)
                elif commandWordList[0] == "exit":
                    safeExit()
                    exit(0)
                else:
                    print("Wrong command")
            except KeyboardInterrupt:
                safeExit()
                exit(0)
            except Exception as e:
                s = traceback.format_exc()
                listS = s.split("\n")[:-1]
                s = "\n".join(listS[-3:])
                print(s)

if __name__ == "__main__":
    main = DLNestLocal()
    main.run()