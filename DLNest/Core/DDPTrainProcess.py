
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import random
import sys
try:
    from .TrainProcess import *
except ImportError:
    from DLNest.Core.TrainProcess import *


class DDPTrainProcess(TrainProcess):
    def __init__(self,task : TrainTask, showOnScreen = False, commandQueue : Queue = None, Port : str = "16484"):
        super(DDPTrainProcess,self).__init__(task,showOnScreen,commandQueue)
        self.port = Port
        self.seed = random.randint(0,2147483647)

    def run(self):
        if isinstance(self.task.GPUID,list):
            ids = [str(item) for item in self.task.GPUID]
            self.card_ids = ids
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
        else:
            print("No enough cards have been specified. DDP needs multi-cards")
            exit(0)
        
        self.__run_unified()

        world_size = len(self.card_ids)
        mp.spawn(
            self.run_seperate,
            nprocs = world_size,
            join = True
        )
    
    def __run_unified(self):
        pass

    def setup_seed(self):
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def __train(self):
        nowEpoch = self.startEpoch
        while True:
            if self.lifeCycle.BOneEpoch() != "Skip":
                self.trainLoader.sampler.set_epoch(nowEpoch)
                for _iter,data in enumerate(self.trainLoader):
                    # run one step
                    if self.lifeCycle.BModelOneStep() != "Skip":
                        self.model.runOneStep(data,self.logDict,_iter,nowEpoch)
                    self.lifeCycle.AModelOneStep()

                    # visualize
                    if self.lifeCycle.needVisualize(nowEpoch,_iter,self.logDict,self.task.args):
                        if self.rank == 0 and self.lifeCycle.BVisualize() != "Skip":
                            self.model.visualize(epoch = nowEpoch, iter = _iter, log = self.logDict)
                        self.lifeCycle.AVisualize()

                # output in command Line
                self.lifeCycle.commandLineOutput(nowEpoch,self.logDict,self.task.args)

                # validation
                if self.lifeCycle.needValidation(nowEpoch,self.logDict,self.task.args):
                    if self.lifeCycle.BValidation() != "Skip":
                        self.model.validate(self.valLoader,self.logDict)
                    self.lifeCycle.AValidation()

                # save checkpoint
                if self.rank == 0:
                    if self.lifeCycle.needSaveModel(nowEpoch,self.logDict,self.task.args):
                        if self.lifeCycle.BSaveModel() != "Skip":
                            self._TrainProcess__saveModel(nowEpoch)
                        self.lifeCycle.ASaveModel()
            
            self.lifeCycle.AOneEpoch()
            # break decision
            if self.lifeCycle.needContinueTrain(nowEpoch,self.logDict,self.task.args):
                nowEpoch += 1
            else:
                break

    def __initOutput(self):
        # init saving root
        saveRoot = Path(self.task.args["save_root"])
        saveDir = saveRoot / self.task.timestamp

        # 重定向输出
        self.outputFile = saveDir / ("_output_" + str(self.rank) + ".txt")
        self.outputFP = self.outputFile.open('w')
        if self.showOnScreen:
            self.outputDelegate = TrainStdout(self.outputFP,True,sys.stdout)
        else:
            self.outputDelegate = TrainStdout(self.outputFP)
        sys.stdout = self.outputDelegate
        sys.stderr = self.outputDelegate

        # redefine the path to src
        self.task.args["root_file_path"] = str(saveDir)

    def run_seperate(self,rank):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.port
        dist.init_process_group("nccl", rank=rank, world_size=len(self.card_ids))

        self.rank = rank
        self.setup_seed()
        self._TrainProcess__loadLifeCycle()
        self._TrainProcess__initLifeCycle()
        self.lifeCycle.trainProcess = self
        self.lifeCycle.rank = self.rank
        if self.lifeCycle.BAll() != "Skip":
            # initialize save folder
            if self.rank == 0 and self.lifeCycle.BSaveInit() != "Skip":
                self._TrainProcess__initSave()
            dist.barrier()
            if self.rank != 0:
                self.__initOutput()    
            self.lifeCycle.ASaveInit()

            self._TrainProcess__loadOthers()

            # initialize dataset
            if self.lifeCycle.BDatasetInit() != "Skip":
                self._TrainProcess__initDataset()
                self.lifeCycle.dataset = self.dataset
            self.lifeCycle.ADatasetInit()

            # initialize model
            if self.lifeCycle.BModelInit() != "Skip":
                self._TrainProcess__initModel()
                self.model.DDPOperation(rank,len(self.card_ids))
                self.lifeCycle.model = self.model
            self.lifeCycle.AModelInit()

            # train
            if self.lifeCycle.BTrain() != "Skip":
                self.__train()
            self.lifeCycle.ATrain()
    
        self.lifeCycle.AAll()
        exit(0)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ta = TrainTask(
        args = {
        "save_root":"/root/code/DLNestDDPTest/Saves",
        "model_name":"Model",
        "dataset_name":"Dataset",
        "life_cycle_name":"LifeCycle",
        "checkpoint_args":{
            "max_ckpt_in_slow_track":100,
            "dilation_in_slow_track":100,
            "max_ckpt_in_fast_track":2,
            "max_ckpt_in_consistent_track":1
        },
        "root_file_path":"/root/code/DLNestDDPTest",
        "model_file_path":"./Model/Model.py",
        "dataset_file_path":"./Dataset/Dataset.py",
        "life_cycle_file_path":"./LifeCycle.py",
        "other_file_paths":[],
        "child_jsons":[
            "./model_config.json",
            "./dataset_config.json"
        ]
        },
        description = "none"
    )
    ta.GPUID = [1,2,3]
    ta.timestamp = "now"
    TP = DDPTrainProcess(ta,showOnScreen=True)
    TP.start()
    print(TP.pid)
    TP.join()