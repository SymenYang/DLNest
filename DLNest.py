from Scheduler import Scheduler
from Train.Task import Task
from Analyze.AnalyzeTask import AnalyzeTask
import json
from pathlib import Path
import argparse
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import shutil

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class DLNestArguments(Arguments):
    def __init__(self):
        super(DLNestArguments, self).__init__(desc="Arguments for DLNest trainner.")

        self._parser.add_argument("-c",type=str,default="./DLNest_config.json",help="Config json for DLNest trainer")

class TaskArguments(Arguments):
    def __init__(self):
        super(TaskArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.")
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-j",type=str, default = "False", help="True for jump in line.(default: False)")
        self._parser.add_argument("-mc",type=str,default = "False", help="True to use multi card if single card can't handle")

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

class DLNest:
    def __init__(self):
        commandArgs = DLNestArguments().parser().parse_args()

        # 载入trainer的参数
        trainerArgsPath = Path(commandArgs.c)
        fp = trainerArgsPath.open("r")
        self.trainerArgs = json.load(fp)
        fp.close()

        # 设置scheduler
        self.scheduler = Scheduler(
            cards = self.trainerArgs["cards"],
            timeDelay = self.trainerArgs["timeDelay"],
            maxTaskPerCard = self.trainerArgs["maxTaskPerCard"]
        )
        self.scheduler.startRoutineTask()
    
        self.taskArgParser = TaskArguments()
        self.projectArgParser = ProjectArguments()
        self.analyzeArgParser = AnalyzeArguments()

    def __replaceArgs(self,newArgName,newArgValue,args):
        """
        newArgName 应该是args下的名称
        若新参数不是一个dict，或新参数不存在覆盖问题，则直接新建该参数或覆盖
        若新参数是一个dict且存在覆盖问题，则递归dict调用
        """
        if not newArgName in args:
            args[newArgName] = newArgValue
            return
        
        if not isinstance(newArgValue,dict):
            args[newArgName] = newArgValue
            return
        
        for key in newArgValue:
            self.__replaceArgs(key,newArgValue[key],args[newArgName])
        return

    def loadArgs(self,filePath : Path,args : dict):
        
        # 若文件不存在或不是一个文件，直接返回
        if not filePath.is_file():
            return False
        try:
            fp = filePath.open('r')
            newArgs = json.load(fp)

            # 对除了child_jsons的每一个key尝试覆盖或新建args
            for key in newArgs:
                if key == "child_jsons":
                    continue
                
                # 为避免dict类型的参数被完全覆盖，使用__replaceArgs方法新建或覆盖args
                self.__replaceArgs(key,newArgs[key],args)

            # 递归查找子json，子json覆盖父json的参数（按照DFS序）
            if "child_jsons" in newArgs:
                for item in newArgs["child_jsons"]:
                    path = Path(item)
                    
                    # 若子json路径不是绝对路径，则按照当前json路径寻找相对路径
                    if not path.is_absolute():
                        path = filePath.parent / item
                    
                    # 载入子json
                    self.loadArgs(path,args)

            fp.close()
            return True
        except Exception as e:
            print(e)
            try:
                fp.close()
            except Exception:
                ...
            return False

    def runTrain(self,commandWordList : list):
        # 获取run命令的参数
        args,otherArgs = self.taskArgParser.parser().parse_known_args(commandWordList[1:])

        # 尝试运行任务
        try:
            taskArgsPath = Path(args.c)
            # 初始化task 参数，包括但不限于模型参数，数据集参数
            taskArgs = {}

            # 若模型参数根json获取失败，报错
            if self.loadArgs(taskArgsPath,taskArgs):
                ...
            else:
                print("Wrong root configuration json file.")
                return

            # 获取高频修改参数
            # 若高频修改参数获取失败，忽略
            if args.f != "":
                hotArgsPath = Path(args.f)
                self.loadArgs(hotArgsPath,taskArgs)
            
            # 获得需要复制的文件名
            modelFilePath = Path(taskArgs["model_file_path"])
            datasetFilePath = Path(taskArgs["dataset_file_path"])
            lifeCycleFilePath = Path(taskArgs["life_cycle_file_path"])
            otherFilePaths = [Path(item) for item in taskArgs["other_file_paths"]]
            # 获取是否需要插队
            jumpInLine = True if args.j == "True" else False
            # 获取是否多卡
            multiCard = True if args.mc == "True" else False
            # 新建Task
            task = Task(
                modelFilePath = modelFilePath,
                datasetFilePath = datasetFilePath,
                lifeCycleFilePath = lifeCycleFilePath,
                otherFilePaths = otherFilePaths,
                args = taskArgs,
                description = args.d,
                memoryConsumption = args.m,
                multiCard=multiCard
            )
            # 运行Task
            self.scheduler.giveATask(task,jumpInLine)
        except Exception as e:
            print(e)
            print("fail to run a task.")
    
    def newProject(self,commandWordList : list):
        args,otherArgs = self.projectArgParser.parser().parse_known_args(commandWordList[1:])
        try:
            projectPath = Path(args.d).absolute()

            # 若目标位置有文件或文件夹，失败退出
            if projectPath.exists():
                print("path Already exists.")
                return

            # 将FactoryFile复制进目标位置
            factoryPath = Path("./FactoryFiles")
            shutil.copytree(factoryPath,projectPath)

            # 修改root_config中的save_root与root_file_path
            rootConfigPath = projectPath / "root_config.json"
            root_config = {}
            with rootConfigPath.open('r') as fp:
                root_config = json.load(fp)
            root_config["save_root"] = str(projectPath / "Saves")
            root_config["root_file_path"] = str(projectPath)
            with rootConfigPath.open('w') as fp:
                json.dump(root_config, fp, indent=4, separators=(',', ':'))

        except Exception as e:
            print(e)
            return

    def analyze(self,commandWordList : list):
        args,otherArgs = self.analyzeArgParser.parser().parse_known_args(commandWordList[1:])
        
        try:
            analyzeTask = AnalyzeTask(
                recordPath = Path(args.r),
                scriptPath = Path(args.s),
                checkpointID = args.c,
                memoryConsumption = args.m
            )
            self.scheduler.giveAnAnalyzeTask(analyzeTask)
        except Exception as e:
            print(e)
            print("fail to run the analyzer.")

    def run(self):
        self.session = PromptSession(auto_suggest=AutoSuggestFromHistory())
        while True:
            command = self.session.prompt(HTML("<seagreen><b>DLNest>></b></seagreen>"))
            commandWordList = command.strip().split(' ')
            if commandWordList[0] == 'run':
                self.runTrain(commandWordList)
            elif commandWordList[0] == 'new':
                self.newProject(commandWordList)
            elif commandWordList[0] == 'analyze':
                self.analyze(commandWordList)
            elif commandWordList[0] == 'show':
                if commandWordList[1] == 'pending':
                    print("[",end='')
                    for item in self.scheduler.pendingTasks:
                        print("Task(" + item.description + ")",end=",")
                    print("]")
            elif commandWordList[0] == 'exit':
                exit(0)
            else:
                print("Use \'run\' to start a new training process, use \'new\' to create a project.")

if __name__ == "__main__":
    tr = DLNest()
    tr.run()