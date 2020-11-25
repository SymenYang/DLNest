import requests
import argparse
import json


class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

def raiseError(self,message):
    raise Exception(message)

class TaskArguments(Arguments):
    def __init__(self):
        super(TaskArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.")
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-j",action='store_true', help="True for jump in line.(default: False)")
        self._parser.add_argument("-mc",action='store_true', help="True to use multi card if single card can't handle")
        self._parser.add_argument("-ns",action='store_true', help="True to save to the NOSAVE dir")

        self._parser.error = raiseError

class AnalyzeArguments(Arguments):
    def __init__(self):
        super(AnalyzeArguments, self).__init__(desc="Arguments for an Analyzer")

        self._parser.add_argument("-r",type=str, help = "path to the model record directory.")
        self._parser.add_argument("-s",type=str, help = "path to the analyze scripts.")
        self._parser.add_argument("-c",type=int, help = "which epoch you want the model to load.(int)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")

        self._parser.error = raiseError

class ProjectArguments(Arguments):
    def __init__(self):
        super(ProjectArguments, self).__init__(desc="Arguments for create a DLNest project.")

        self._parser.add_argument("-d",type=str, help="Path to the directory you want to create the project.")
        self._parser.error = raiseError

class CardChangeArguments(Arguments):
    def __init__(self):
        super(CardChangeArguments, self).__init__(desc="Arguments for change valid cards.")
        self._parser.add_argument("-c",type=int, default=[0,1,2,3], nargs='+', help='valid cards')

class CommandCommunicator:
    def __init__(self,url : str):
        self.url = url
        self.taskArgParser = TaskArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()
        self.cardChangeArgParser = CardChangeArguments()
        self.app = None
    
    def runTrain(self,commandWordList : list):
        newList = []
        now = ""
        for item in commandWordList:
            if item[0] == "-": 
                newList.append(now)
                now = ""
                newList.append(item)
                continue
            else:
                if now == "":
                    now = now + item
                else:
                    now = now + " " + item
        if now != "": 
            newList.append(now)
        # 获取run命令的参数

        try:
            args,otherArgs = self.taskArgParser.parser().parse_known_args(newList[1:])
            r = requests.post(self.url + "/run_train",{
                "root_config" : args.c,
                "description" : args.d,
                "freq_config" : args.f,
                "memory_consumption" : args.m,
                "jump_in_line" : args.j,
                "multi_card" : args.mc,
                "no_save" : args.ns
            })
            return r
        except Exception:
            return None

    def newProject(self,commandWordList : list):
        try:
            args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
            r = requests.post(self.url + "/new_proj",{
                "target_dir" : args.d
            })
            return r
        except Exception:
            return None

    def runAnalyze(self,commandWordList : list):
        try:
            args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
            r = requests.post(self.url + "/load_model",{
                "record_path" : args.r,
                "script_path" : args.s,
                "checkpoint_ID" : args.c,
                "memory_consumption" : args.m
            })
            return r
        except Exception:
            return None

    def changeCards(self,commandWordList : list):
        try:
            args,otherArgs = self.cardChangeArgParser.parser().parse_known_args(commandWordList[1:])
            r = requests.post(self.url + "/change_valid_cards",{
                "cards" : args.c
            })
            return r
        except Exception as e:
            print(e)
            return None

    def runExp(self,command : str):
        try:
            r = requests.post(self.url + "/run_exp",{
                "command" : command
            })
            return r
        except Exception:
            return None
    
    def release(self):
        try:
            r = requests.get(self.url + "/release_model")
            return r
        except Exception:
            return None

    def kill(self,taskID : str):
        try:
            r = requests.post(self.url + "/del_task",{
                "task_ID" : taskID
            })
            return r
        except Exception:
            return None

    def giveACommand(self,command : str):
        commandWordList = command.strip().split(' ')
        if commandWordList[0] == 'run':
            return self.runTrain(commandWordList)
        elif commandWordList[0] == 'new':
            return self.newProject(commandWordList)
        elif commandWordList[0] == 'load':
            return self.runAnalyze(commandWordList)
        elif commandWordList[0] == 'runExp':
            if len(commandWordList) < 2:
                return "wrong command"
            return self.runExp(commandWordList[1])
        elif commandWordList[0] == 'release':
            return self.release()
        elif commandWordList[0] == 'del':
            return self.kill(commandWordList[1])
        elif commandWordList[0] == 'exit':
            self.app.exit()
        elif commandWordList[0] == 'changeCards':
            self.changeCards(commandWordList)