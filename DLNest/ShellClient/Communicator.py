import requests
import argparse
import traceback
import json
from functools import wraps

def raiseError(self):
    raise Exception("command error")

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)
        self._parser.error = raiseError

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

class OutputArguments(Arguments):
    def __init__(self):
        super(OutputArguments, self).__init__(desc="Argumetns for get styled or not styled outputs.")
        self._parser.add_argument("-t",type=str, default = "", help="task ID")
        self._parser.add_argument("-s",action="store_true",help = "set to get styled")

def stableRun(f):
    wraps(f)
    def doStableRun(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            return {"status" : "error", "error" : str(e)}
    return doStableRun


class Communicator:
    def __init__(self, url : str = "127.0.0.1", port : int = 9999):
        self.trainArgParser = TrainArguments()
        self.continueArgParser = ContinueArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()
        self.deviceChangeArgParser = DeviceChangeArguments()
        self.outputArgParser = OutputArguments()
        self.url = "http://" + url + ":" + str(port)

    def shortenList(self,commandWordList : list):
        newList = []
        now = ""
        for item in commandWordList:
            if item[0] == "-": 
                if now != "":
                    newList.append(now)
                now = ""
                newList.append(item)
                continue
            elif item[0] == "\"":
                if now != "":
                    newList.append(now)
                now = item
            else:
                if now == "":
                    now = now + item
                else:
                    now = now + " " + item
        if now != "": 
            newList.append(now)
        for i in range(len(newList)):
            item = newList[i]
            if item[0] == "\"" and item[-1] == "\"":
                newList[i] = newList[i][1:-1]
        return newList

    @stableRun
    def runTrain(self,commandWordList : list):
        commandWordList = self.shortenList(commandWordList)
        args,otherArgs = self.trainArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/run_train",{
            "config_path" : args.c,
            "freq_path" : args.f,
            "description" : args.d,
            "memory_consumption" : args.m,
            "CPU" : args.CPU,
            "DDP" : args.DDP,
            "multi_GPU" : args.mc,
            "no_save" : args.ns,
            "use_description" : args.sd,
        })
        return json.loads(r.text)

    @stableRun
    def newProject(self,commandWordList : list):
        commandWordList = self.shortenList(commandWordList)
        args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/new_proj",{
            "target_dir" : args.d,
            "MNIST" : args.MNIST
        })
        return json.loads(r.text)

    @stableRun
    def runAnalyze(self,commandWordList : list):
        commandWordList = self.shortenList(commandWordList)
        args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/analyze",{
            "record_path" : args.r,
            "script_path" : args.s,
            "checkpoint_ID" : args.c,
            "CPU" : args.CPU,
            "memory_consumption" : args.m
        })
        return json.loads(r.text)

    @stableRun
    def continueTrain(self,commandWordList : list):
        commandWordList = self.shortenList(commandWordList)
        args,otherArgs = self.continueArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/continue_train",{
            "record_path" : args.r,
            "checkpoint_ID" : args.c,
            "memory_consumption" : args.m,
            "CPU" : args.CPU,
            "DDP" : args.DDP,
            "multi_GPU" : args.mc,
            "description" : args.d
        })
        return json.loads(r.text)

    @stableRun
    def changeDevices(self,commandWordList : list):
        args,otherArgs = self.deviceChangeArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.post(self.url + "/change_devices",{
            "new_devices_IDs" : args.d
        })
        return json.loads(r.text)

    @stableRun
    def runExp(self,commandWordList : list):
        """
        commandWordList : taskID , command
        """
        r = requests.post(self.url + "/run_exp",{
            "task_ID" : commandWordList[1],
            "command" : commandWordList[2]
        })
        return json.loads(r.text)

    @stableRun
    def getAnalyzeOutput(self,commandWordList : list):
        """
        commandWordList : -t  task_ID -s
        """
        args,otherArgs = self.outputArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.get(self.url + "/get_analyze_output",{
            "task_ID" : args.t,
            "styled" : args.s
        })
        return json.loads(r.text)

    @stableRun
    def getDLNestOutput(self,commandWordList : list):
        """
        commandWordList : -s
        """
        args,otherArgs = self.outputArgParser.parser().parse_known_args(commandWordList[1:])
        r = requests.get(self.url + "/get_DLNest_output",{
            "styled" : args.s
        })
        return json.loads(r.text)
    
    @stableRun
    def getTasksInformation(self):
        r = requests.get(self.url + "/get_task_info",{})
        return json.loads(r.text)

    @stableRun
    def getDevicesInformation(self):
        r = requests.get(self.url + "/get_devices_info",{})
        return json.loads(r.text)

    @stableRun
    def delATask(self,commandWordList : list):
        """
        commandWordList: taskID
        """
        r = requests.post(self.url + "/del_task",{
            "task_ID" : commandWordList
        })
        return json.loads(r.text)

    @stableRun
    def clear(self):
        r = requests.post(self.url + "/clear",{})
        return json.loads(r.text)

    def giveACommand(self, commandWordList : list):
        if commandWordList[0] == "run":
            return self.runTrain(commandWordList)
        elif commandWordList[0] == "continue":
            return self.continueTrain(commandWordList)
        elif commandWordList[0] == "new":
            return self.newProject(commandWordList)
        elif commandWordList[0] == "analyze":
            return self.runAnalyze(commandWordList)
        elif commandWordList[0] == "runExp":
            return self.runExp(commandWordList)
        elif commandWordList[0] == "del":
            return self.delATask(commandWordList[1])
        elif commandWordList[0] == "showAN":
            return self.getAnalyzeOutput(commandWordList)
        elif commandWordList[0] == "showDL":
            return self.getDLNestOutput(commandWordList)
        elif commandWordList[0] == "showTask":
            return self.getTasksInformation()
        elif commandWordList[0] == "showDevice":
            return self.getDevicesInformation()
        elif commandWordList[0] == "changeDevices":
            return self.changeDevices(commandWordList)
        elif commandWordList[0] == "clear":
            return self.clear()
        elif commandWordList[0] == "exit":
            return {"exit" : True}
        else:
            return {"error" : "Wrong command"}