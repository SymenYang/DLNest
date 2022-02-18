import shutil
import json
from pathlib import Path
import time
import random
try:
    import torch
except ImportError:
    # need to costomize the doSaveOperation and doLoadOperation functions
    pass

class SavePackage:
    def __init__(
        self,
        configPath : str = "",
        freqPath : str = ""
    ):
        self.args = {}
        haveArgs = False
        if configPath != "":
            configPath = Path(configPath)
            self.__loadArgs(configPath,self.args)
            haveArgs = True
        if freqPath != "":
            freqPath = Path(freqPath)
            self.__loadArgs(freqPath,self.args)
            haveArgs = True
        
        self.ckpts = {}
        self.nowCkptID = 0
        self.ckptSlow = []
        self.ckptFast = []
        self.ckptConsistent = []
        self.checkpointsDir = None
        if haveArgs:
            self.maxCkptSlow = self.args["checkpoint_args"]["max_ckpt_in_slow_track"]
            self.maxCkptFast = self.args["checkpoint_args"]["max_ckpt_in_fast_track"]
            self.maxCkptConsisitent = self.args["checkpoint_args"]["max_ckpt_in_consistent_track"]
            self.slowDilation = self.args["checkpoint_args"]["dilation_in_slow_track"]

        self.root = None
        self.checkpointsDir = None
        self.prefix = "state_"

    def giveArgs(self,args : dict): # Not used
        self.args = args
        self.maxCkptSlow = self.args["checkpoint_args"]["max_ckpt_in_slow_track"]
        self.maxCkptFast = self.args["checkpoint_args"]["max_ckpt_in_fast_track"]
        self.maxCkptConsisitent = self.args["checkpoint_args"]["max_ckpt_in_consistent_track"]
        self.slowDilation = self.args["checkpoint_args"]["dilation_in_slow_track"]

    def __replaceArgs(self,newArgName,newArgValue,args):
        """
        newArgName : 应该是args下的名称
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

    def __loadArgs(self,filePath : Path,args : dict):
        """
        load configs to the args dict
        """
        # 若文件不存在或不是一个文件，直接报错
        if not filePath.is_file():
            raise BaseException("Config file doesn't exists. " + str(filePath))
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
                self.__loadArgs(path,args)

        fp.close()
        return
    
    def __copyAFile(self,filePath : Path, saveDir : Path):
        '''
        若filePath为相对路径，则复制到其对应文件夹
        若filePath为绝对路径，则复制到储存包的根
        '''
        if filePath.is_absolute():
            if filePath.is_dir():
                shutil.copytree(filePath,saveDir / filePath.stem)
            else:
                shutil.copy(filePath,saveDir / (filePath.stem + filePath.suffix))
        else:
            abFilePath = Path(self.args["root_file_path"]) / filePath
            if abFilePath.is_dir():
                shutil.copytree(abFilePath,saveDir / filePath)
            else:
                target = saveDir / filePath
                target_dir = target.parent
                if not target_dir.exists():
                    target_dir.mkdir(parents = True, exist_ok = True)
                shutil.copy(abFilePath,target)

    def __savePackageInformation(self):
        packageInfoPath = self.root / "_package.json"
        packageInfo = {
            "ckpt_slow" : self.ckptSlow,
            "ckpt_fast" : self.ckptFast,
            "ckpt_consistent" : self.ckptConsistent,
            "prefix" : self.prefix
        }
        with packageInfoPath.open("w") as fp:
            json.dump(packageInfo,fp,sort_keys=True, indent=4, separators=(',', ':'))

    def __loadPackageInformation(self):
        packageInfoPath = self.root / "_package.json"
        packageInfo = {}
        with packageInfoPath.open("r") as fp:
            packageInfo = json.load(fp)
        self.ckptSlow = packageInfo["ckpt_slow"]
        self.ckptFast = packageInfo["ckpt_fast"]
        self.ckptConsistent = packageInfo["ckpt_consistent"]
        self.prefix = packageInfo["prefix"]

    def saveToNewDir(self,overrideSaveName : str = "",copyFiles = True):
        """
        Make a new save package. If having the override save name, use it. If not, use timestamp to save.
        """
        saveRoot = Path(self.args["save_root"])
        if not saveRoot.is_absolute():
            saveRoot = Path(self.args["root_file_path"]) / saveRoot
            self.args["save_root"] = str(saveRoot)
        nowTime = time.time()
        saveName = ""
        if overrideSaveName != "":
            saveName = overrideSaveName
        else:
            saveName = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(nowTime)) + "_" + str(random.randint(100,999)) # avoid conflict. May be changed later
        saveDir = saveRoot / saveName
        self.root = saveDir

        if saveDir.exists():
            shutil.rmtree(saveDir)
        saveDir.mkdir(parents=True,exist_ok=True)

        #checkpoints save dir
        self.checkpointsDir = saveDir / "Checkpoints"
        self.checkpointsDir.mkdir(parents=True,exist_ok=True)
        self.ckpts = {}

        # copy python files into dir if copy Files is TRUE
        if copyFiles:
            self.__copyAFile(Path(self.args["runner_file_path"] if "runner_file_path" in self.args else self.args["model_file_path"]),saveDir) # need to be deprecated
            self.__copyAFile(Path(self.args["dataset_file_path"]),saveDir)
            self.__copyAFile(Path(self.args["life_cycle_file_path"]),saveDir)
            for item in self.args["other_file_paths"]:
                self.__copyAFile(Path(item),saveDir)

        # save args
        argsPath = saveDir / "args.json"
        argsFP = argsPath.open('w')
        self.args["root_file_path"] = str(saveDir)
        json.dump(self.args,argsFP,sort_keys=True, indent=4, separators=(',', ':'))
        argsFP.close()

        # save package info
        self.__savePackageInformation()

    def saveACheckpoint(self,stateDict : dict, holdThisCheckpoint : bool = False):
        """
        Save a new checkpoint and manage the storage.
        """
        idsNeed2Delete = []

        # save this state dict
        saveFile = self.checkpointsDir / (self.prefix + str(self.nowCkptID) + ".ckpt")
        saveName = str(saveFile)
        self.doSaveOperation(stateDict,saveName)

        # add to ckpts dict
        self.ckpts[self.nowCkptID] = saveFile
        
        # append in fast track
        self.ckptFast.append(self.nowCkptID)
        if len(self.ckptFast) > self.maxCkptFast:
            w2did = self.ckptFast.pop(0)
            idsNeed2Delete.append(w2did)
        
        # append in slow track
        if self.nowCkptID % self.slowDilation == 0:
            self.ckptSlow.append(self.nowCkptID)
            if len(self.ckptSlow) > self.maxCkptSlow:
                w2did = self.ckptSlow.pop(0)
                if not w2did in idsNeed2Delete:
                    idsNeed2Delete.append(w2did)
        
        # append in consistent track
        if holdThisCheckpoint:
            self.ckptConsistent.append(self.nowCkptID)
            if len(self.ckptConsistent) > self.maxCkptConsisitent:
                w2did = self.ckptConsistent.pop(0)
                if not w2did in idsNeed2Delete:
                    idsNeed2Delete.append(w2did)
        
        #delete useless checkpoints on disk
        for id in idsNeed2Delete:
            if not (id in self.ckptFast or
                    id in self.ckptSlow or
                    id in self.ckptConsistent):
                path = self.ckpts[id]
                path.unlink()
                self.ckpts.pop(id)
        
        self.nowCkptID += 1
        # save package info
        self.__savePackageInformation()
    
    def doSaveOperation(self,stateDict,fileName):
        """
        pytorch version of save ckpt. If no torch in the environment, please override this function.
        """
        torch.save(stateDict,fileName)

    def doLoadOperation(self,fileName : str,device):
        return torch.load(fileName,map_location = device)

    def saveVisualString(self, visualString : str):
        """
        Make a new file with this visual string which makes the information noticeable.
        """
        visualFile = self.root / ("_" + visualString)
        visualFile.touch()
        with open(visualFile,"w") as f:
            f.write(visualString)

    def initCkptsFromExist(self):
        self.checkpointsDir = self.root / "Checkpoints"
        filelist = self.checkpointsDir.iterdir()
        for item in filelist:
            if item.suffix == ".ckpt":
                id = int(item.stem.split("_")[-1])
                self.ckpts[id] = item
                self.nowCkptID = max(self.nowCkptID,id)
        self.nowCkptID += 1

    def initFromAnExistSavePackage(self,packagePath : str):
        self.root = Path(packagePath)
        configPath = self.root / "args.json"
        self.args = {}
        self.__loadArgs(configPath,self.args)
        
        self.maxCkptSlow = self.args["checkpoint_args"]["max_ckpt_in_slow_track"]
        self.maxCkptFast = self.args["checkpoint_args"]["max_ckpt_in_fast_track"]
        self.maxCkptConsisitent = self.args["checkpoint_args"]["max_ckpt_in_consistent_track"]
        self.slowDilation = self.args["checkpoint_args"]["dilation_in_slow_track"]

        # load package info
        self.__loadPackageInformation()

        # load ckpt
        self.initCkptsFromExist()

    def getStateDict(self,id = -1,device = "cuda:0"):
        if id == -1:
            id = self.nowCkptID - 1
        if not id in self.ckpts:
            raise BaseException("The checkpoint doesn't exist.")
        return self.doLoadOperation(str(self.ckpts[id]),device)

    def getOutputFile(self,rank = -1):
        outputFileName = "_output.txt"
        if rank != -1:
            outputFileName = "_output_" + str(rank) + ".txt" 
        outputFilePath = self.root / outputFileName
        if outputFilePath.exists():
            return outputFilePath.open("a")
        else:
            return outputFilePath.open("w")

    def setCkptID(self,ckptID : int):
        self.nowCkptID = ckptID
        
        newslow = []
        for item in self.ckptSlow:
            if item >= self.nowCkptID:
                pass
            else:
                newslow.append(item)
        self.ckptSlow = newslow
        
        newfast = []
        for item in self.ckptFast:
            if item >= self.nowCkptID:
                pass
            else:
                newfast.append(item)
        self.ckptFast = newfast
        
        newconsistent = []
        for item in self.ckptConsistent:
            if item >= self.nowCkptID:
                pass
            else:
                newconsistent.append(item)
        self.ckptConsistent = newconsistent

        popList = []
        for key in self.ckpts:
            if key >= self.nowCkptID:
                popList.append(key)
        for item in popList:
            self.ckpts.pop(item)