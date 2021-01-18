import json
import argparse
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import requests


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
        self._parser.add_argument("-s",type=str, default="", help = "path to the analyze scripts.")
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

class DLNestSimpleClient:
    def __init__(self,url : str):
        self.url = url
        self.taskArgParser = TaskArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()
        self.cardChangeArgParser = CardChangeArguments()
    
    def runTrain(self,commandWordList : list):
        # 获取run命令的参数
        args,otherArgs = self.taskArgParser.parser().parse_known_args(commandWordList[1:])

        r = requests.post(self.url + "/run_train",{
            "root_config" : args.c,
            "description" : args.d,
            "freq_config" : args.f,
            "memory_consumption" : args.m,
            "jump_in_line" : True if args.j == "True" else False,
            "multi_card" : True if args.mc == "True" else False,
            "no_save" : True if args.ns == "True" else False
        })
        print(r.content)
    
    def newProject(self,commandWordList : list):
        args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/new_proj",{
            "target_dir" : args.d
        })
        print(r.content)

    def runAnalyze(self,commandWordList : list):
        args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/load_model",{
            "record_path" : args.r,
            "script_path" : args.s,
            "checkpoint_ID" : args.c,
            "memory_consumption" : args.m
        })
        print(r.content)

    def changeCards(self,commandWordList : list):
        args,otherArgs = self.cardChangeArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/change_valid_cards",{
            "cards" : args.c
        })
        print(r.content)

    def runExp(self,command : str):
        r = requests.post(self.url + "/run_exp",{
            "command" : command
        })
        print(r.content)
    
    def release(self):
        r = requests.get(self.url + "/release_model")
        print(r.content)

    def kill(self,taskID : str):
        r = requests.post(self.url + "/del_task",{
            "task_ID" : taskID
        })
        print(r.content)

    def suspend(self,taskID : str):
        r = requests.post(self.url + "/sus_task",{
            "task_ID" : taskID
        })
        print(r.content)

    def reload(self,taskID : str):
        r = requests.post(self.url + "/reload_task",{
            "task_ID" : taskID
        })
        print(r.content)

    def showDL(self):
        r = requests.get(self.url + "/DLNest_buffer")
        print(json.loads(r.content)["text"])
    
    def showAN(self):
        r = requests.get(self.url + "/analyzer_buffer")
        print(json.loads(r.content)["text"])

    def taskInfo(self):
        r = requests.get(self.url + "/task_info")
        print(json.loads(r.content)["info"])

    def showCards(self):
        r = requests.get(self.url + "/cards_info")
        print(r.content)

    def changeTimeDelay(self,delay):
        try:
            r = requests.post(self.url + "/change_time_delay",{
                "delay" : int(delay)
            })
            print(r.content)
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
                self.runExp(commandWordList[1])
            elif commandWordList[0] == 'release':
                self.release()
            elif commandWordList[0] == 'del':
                self.kill(commandWordList[1])
            elif commandWordList[0] == 'suspend':
                self.suspend(commandWordList[1])
            elif commandWordList[0] == 'reload':
                self.reload(commandWordList[1])
            elif commandWordList[0] == 'showDL':
                self.showDL()
            elif commandWordList[0] == 'showAN':
                self.showAN()
            elif commandWordList[0] == 'show':
                self.taskInfo()
            elif commandWordList[0] == 'exit':
                exit(0)
            elif commandWordList[0] == 'changeCards':
                self.changeCards(commandWordList)
            elif commandWordList[0] == 'showCards':
                self.showCards()
            elif commandWordList[0] == "changeDelay":
                self.changeTimeDelay(commandWordList[1])
            else:
                print("Use \'run\' to start a new training process, use \'new\' to create a project.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u",type=str, default="http://127.0.0.1:9998",help="DLNest server address")
    args=parser.parse_args()
    url = args.u
    if url[:7] != "http://":
        url = "http://" + url
    main = DLNestSimpleClient(url)
    main.run()