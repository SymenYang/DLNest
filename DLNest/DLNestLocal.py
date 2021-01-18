from .Core.DLNestCore import DLNestCore
import json
from pathlib import Path
import argparse
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory


class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class TaskArguments(Arguments):
    def __init__(self):
        super(TaskArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.")
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-j",type=str, default = "False", help="True for jump in line.(default: False)")
        self._parser.add_argument("-mc",type=str,default = "False", help="True to use multi card if single card can't handle")
        self._parser.add_argument("-ns",type=str,default = "False", help="True to use multi card if single card can't handle")

class AnalyzeArguments(Arguments):
    def __init__(self):
        super(AnalyzeArguments, self).__init__(desc="Arguments for an Analyzer")

        self._parser.add_argument("-r",type=str, help = "path to the model record directory.")
        self._parser.add_argument("-s",type=str, help = "path to the analyze scripts.")
        self._parser.add_argument("-c",type=int, help = "which epoch you want the model to load.(int)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")

class ProjectArguments(Arguments):
    def __init__(self):
        super(ProjectArguments, self).__init__(desc="Arguments for create a DLNest project.")

        self._parser.add_argument("-d",type=str, help="Path to the directory you want to create the project.")

class CardChangeArguments(Arguments):
    def __init__(self):
        super(CardChangeArguments, self).__init__(desc="Arguments for change valid cards.")
        self._parser.add_argument("-c",type=int, default=[0,1,2,3], nargs='+', help='valid cards')


class DLNestDEBUG:
    def __init__(self):
        self.core = DLNestCore()
        self.taskArgParser = TaskArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()
        self.cardChangeArgParser = CardChangeArguments()
    
    def runTrain(self,commandWordList : list):
        # 获取run命令的参数
        args,otherArgs = self.taskArgParser.parser().parse_known_args(commandWordList[1:])

        self.core.runTrain(
            rootConfig=args.c,
            description=args.d,
            freqConfig=args.f,
            memoryConsumption=args.m,
            jumpInLine=True if args.j == "True" else False,
            multiCard=True if args.mc == "True" else False,
            noSave=True if args.ns == "True" else False
        )
    
    def newProject(self,commandWordList : list):
        args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
        self.core.newProj(args.d)

    def runAnalyze(self,commandWordList : list):
        args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
        self.core.loadModel(
            recordPath=args.r,
            scriptPath=args.s,
            checkpointID=args.c,
            memoryConsumption=args.m
        )

    def changeCards(self,commandWordList : list):
        args,otherArgs = self.cardChangeArgParser.parser().parse_known_args(commandWordList[1:])
        try:
            self.core.changeCards(args.c)
        except Exception as e:
            print(e)

    def run(self):
        self.session = PromptSession(auto_suggest=AutoSuggestFromHistory())
        while True:
            command = self.session.prompt(HTML("<seagreen><b>DLNest>></b></seagreen>"))
            commandWordList = command.strip().split(' ')
            if commandWordList[0] == 'run':
                self.runTrain(commandWordList)
            elif commandWordList[0] == 'new':
                self.newProject(commandWordList)
            elif commandWordList[0] == 'load':
                self.runAnalyze(commandWordList)
            elif commandWordList[0] == 'runExp':
                if len(commandWordList) < 2:
                    continue
                self.core.runExp(commandWordList[1])
            elif commandWordList[0] == 'release':
                self.core.releaseModel()
            elif commandWordList[0] == 'del':
                self.core.delTask(commandWordList[1])
            elif commandWordList[0] == 'suspend':
                self.core.susTask(commandWordList[1])
            elif commandWordList[0] == 'reload':
                self.core.reloadTask(commandWordList[1])
            elif commandWordList[0] == 'showDL':
                print(self.core.getDLNestOutput()[1])
            elif commandWordList[0] == 'showAN':
                print(self.core.getAnalyzerOutput()[1])
            elif commandWordList[0] == 'show':
                print(self.core.getTasks())
            elif commandWordList[0] == 'exit':
                self.core.releaseModel()
                exit(0)
            elif commandWordList[0] == 'showCard':
                print(self.core.getCardsInfo())
            elif commandWordList[0] == 'changeCards':
                self.changeCards(commandWordList)
            else:
                print("Use \'run\' to start a new training process, use \'new\' to create a project.")

if __name__ == "__main__":
    main = DLNestDEBUG()
    main.run()