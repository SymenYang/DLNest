# DLNest
**DLNest处在开发初期，功能可能发生巨大变化，可能有严重bug，请谨慎使用**  
DLNest是一个深度学习训练\实验框架，轻依赖于pytorch，实现单机自动排卡，训练任务列表，高可复现自动记录，自动载入模型进行实验等功能
## 快速上手
### 安装
安装DLNest：
```bash
git clone https://github.com/SymenYang/DLNest.git
cd DLNest
pip install .
```
启动DLNest服务器：
```bash
python -m DLNest.Server
```
维持服务器运行的同时，启动Shell前端：
```bash
python -m DLNest.Client
```
使用new命令新建一个DLNest的项目，在快速上手教程中使用-MNIST选项新建一个完整的MNIST项目
```bash
new -d <项目新建绝对路径> -MNIST
```
可以看到在项目新建路径内，目录结构如下：
```
|--<Project Dir>
    |--AnalyzeScripts
    |   |--getAcc.py
    |   |--showModel.py
    |--Dataset
    |   |--Dataset.py
    |--Model
    |   |--MNISTCNN.py
    |   |--Model.py
    |--common_config.json
    |--dataset_config.json
    |--freq_config.json
    |--model_config.json
    |--root_config.json
    |--LifeCycle.py
```
在客户端中，使用以下命令即可简单的开始训练：
```
run -c <到root_config的绝对路径>
```
现在，DLNest会开始一次训练，并将训练所用的信息保存在```\<Project Dir\>/Saves/运行时间```的目录内，称为一个保存包。保存包保存的内容包括运行时的代码、所有的参数、所有的checkpoints、训练的输出等。训练输出保存在_output.txt内  
使用如下命令能够终止训练：
```
del <TaskID>
```
TaskID显示在Client下部的任务列表内，通常以"A_"或"T_"开头  
使用如下命令能够继续一个未完成的训练：
```
continue -r <到保存包的绝对路径>
```
该命令能够自动从最近的checkpoints，使用保存包内的代码进行训练。继续训练的模型输出会接在之前的模型输出之后  
在训练完成后，可以使用以下命令来加载一个保存包进入分析器：
```
analyze -r <到保存包的绝对路径>
```
加载的分析任务会成为一个独立任务，具有一个TaskID。想要看到该分析任务的输出，可使用以下命令：
```
watch <TaskID>
```
通过这个命令能够将分析任务的输出加载到客户端的分析器输出窗口。这个输出是随时更新的。  
使用以下命令运行一个实验（以showModel为例）：
```
runExp showModel
```
这个命令会在分析器内运行```\<Project Dir\>/AnalyzeScripts/showModel.py```代码中的脚本，并在分析器输出中进行输出。
使用如下命令将现有分析任务推出分析器显示：
```
withdraw
```
该命令并不会结束这个分析任务，对于在后台的分析任务，可以使用以下命令对其运行分析脚本（以showModel为例）：
```
runExp TaskID showModel
```
结束分析任务依然使用del命令进行。 
在训练时，能够自动使用pytorch的DistributedDataParallel功能进行多卡同步训练（需要模型本身是pytorch模型），使用如下命令进行：
```
run -c <到root_config的绝对路径> -DDP -m <预估的显存占用，需要大于一张卡的空余显存才会分配多张卡>
```
使用DDP进行训练，不同进程的输出会保存在"_output\<rank\>.txt"内
同时，训练、分析也能够使用CPU与内存进行，并不需要显卡，如下所示：
```
# 使用CPU训练
run -c <到root_config的绝对路径> -CPU
# 使用CPU集训训练
continue -r <到保存包的绝对路径> -CPU
# 使用CPU进行分析
analyze -r <到保存包的绝对路径> -CPU
```
## DLNest架构
DLNest进行训练和分析的实现中，以下几个概念较为重要：1. 保存包，2. 训练/分析任务信息， 3. 训练/分析任务进程， 4. 信息中心， 5. 调度器。其各自的作用分别是：
1. 保存包：解析入口配置文件（root_config.json），保存训练时所有参数与代码，构建一个能够直接复现运行的环境；保存checkpoints，并提供较为方便的加载、自动清理无用checkpoints、恢复等功能；提供训练时输出重定向文件。
2. 训练/分析任务信息：包含一个训练/分析任务所需要的所有信息，包括任务ID、保存包、加载的checkpoints编号、现在状态（运行或等待）、运行的子进程实例（如果在运行状态）、预估显存占用、使用的设备、是否使用多卡或DDP等配置信息。其中训练任务信息额外包括新建保存包的相关配置，分析任务信息额外包括一个用于多进程通讯的命令队列和输出类。
3. 训练/分析任务进程：接受一个训练/分析任务信息，新建子进程进行训练/分析任务；根据任务信息配置设备、DDP等；控制训练和分析任务的所有流程；自动重定向训练和分析任务的输出；定时在保存包中保存checkpoint等。
4. 信息中心：保存现有的所有任务信息、保存和更新现有的所有设备信息、保存可用设备列表、提供设备、任务信息的线程安全访问和修改
5. 调度器：每次分析从信息中心中获取设备与任务信息，按给定的调度器策略安排任务运行，包括判断是否运行任务与指定运行设备等。调度器的判断在被呼叫时和一定时间间隔时自动执行。  

在这些类之上，封装了DLNest能够进行的操作为Operation层。Operation层主要有以下几个功能：  
1. 得到信息，包括任务信息、设备信息
2. 得到输出，包括主进程输出与分析进程输出，训练进程输出直接重定向进入保存包指定的文件
3. 通过调度器进行训练/分析任务，得到一些任务配置，在信息中心中加入该等待任务，并呼叫一次调度器的调度判断
4. 直接进行训练/分析任务，得到一些任务配置，不通过调度器与信息中心，直接运行训练/分析任务。直接运行仅用在单次分析与单次训练中
5. 给指定分析任务运行分析脚本
6. 更改DLNest配置，包括调度器运行间隔时间、可用设备列表、单设备最大任务数等
7. 新建项目  

Server则通过调用Operation层中的函数来进行各式各样的操作。
这些概念中，与训练、分析密切相关且用户较常接触的的主要是保存包与训练/分析任务进程。下面对这两个部分进行讲解
### 保存包
保存包的设计目标是能够帮助高可复现的训练进行，并在此基础上方便的进行分析与重训练。因此我们设计了保存包的文件概念与python类，并将其作为DLNest的基础。同时，保存包类也能够独立使用，与DLNest的其他部分进行了充分的解耦。  
为了实现高可复现的目标，许多工具使用自动化git工具来保存代码等。但这样的实现存在几个问题，包括影响git中的commit记录，难以同时分析不同时间的训练结果等。我们发现，相较于保存的Checkpoint文件每一个动辄数十M的大小，所有代码文件加起来的大小可以说微乎其微，且除了代码以外训练中其他的部分很少进行改动（例如数据集文件）。因此我们决定直接拷贝训练时的所有代码，再在拷贝之后的保存包中运行训练，相当于每一个保存包都包含了一个完整的项目。相较于运行后再拷贝，拷贝后再运行能够在命令给出的很短时间内隔离出当次训练的环境，给任务排队等待的实现留出了空间，避免不同次训练间的代码、参数互相影响。  
保存包的文件分布大致如下：
```
|--<Project Dir>
    |--Checkpoints
    |   |--state_x.ckpt
    |   |--state_y.ckpt
    |--Dataset
    |   |--Dataset.py
    |--Model
    |   |--Model.py
    |--LifeCycle.py
    |--args.json
    |--_package.json
    |--_output.txt
```
可以看到与项目目录格式基本一致，区别在于多了Checkpoints目录与一些json文件。其中。args.json为训练时所有参数的集合，相当于所有参数都写在了root_config.json中的效果。由于相似的目录结构，DLNest能够很方便的从一个保存包中再次加载模型和数据集，进行分析、继续训练或者重新训练。  
\_package.json保存了保存包加载所需要的一些信息，包括checkpoints前缀名（默认为“state\_”）、不同保存track保存的index等。DLNest的保存包使用快、慢、持久三通道保存策略，其中快通道保存最新的一定数量checkpoint，慢通道每隔一定间隔保存一个checkpoint，而持久通道则由用户代码来定义是否进入。每个通道都有一个checkpoint数量上限，当该通道checkpoint数量超过上限时，最老的checkpoints将会被从硬盘中删除。由于一个checkpoint能够进入多个通道，因此一个checkpoint只会在其被所有通道抛弃之后才被从硬盘里删除。训练结束之后也能够通过\_package.json来辨认不同通道的checkpoint的index分别为多少。  
\_output.txt则保存了训练进程的输出。若训练进程有多个，则每个进程一个输出文件。
### 训练/分析任务进程
训练任务进程控制了整个训练的流程，其输入为一个训练任务信息，包括了保存包、训练设备配置等内容。而分析进程则加载一个保存包进行分析，运行分析脚本。这两种进程有很大一部分工作是相同的，包括加载代码等，其流程如下：
1. 判断设备信息，包括CPU、GPU、GPUs、DDP（仅训练）
2. 如果是DDP，则构建DDP环境启动DDP进程，启动后每个进程的运行内容与不适用DDP相同
3. 调用训练或分析进程的输出初始化函数，在训练进程中每个训练进程得到一个输出文件描述符，并将stdout和stderr重定向到该文件。在分析进程中，则将输出重定向到AnalyzerBuffer类，进行跨进程的通信。
4. 调用训练、分析进程的主循环  

其中，训练进程的主循环流程如下：
1. 从训练数据集loader中获取数据（enumrate）
2. 根据设备情况将数据分配到对的设备（GPU中）
3. 用该数据调用模型的runOneStep函数
4. 根据lifeCycle的needVisualize函数返回结果，调用或不调用模型的visualize函数。其中在DDP的环境下，只有rank为0的进程会被调用visualize函数
5. 若数据集loader中还有数据（未完成一个epoch）则返回1，若完成了一个epoch，则继续往下
6. 调用lifeCycle中的commandLineOutput函数
7. 根据lifeCycle的needValidation函数返回结果，调用或不调用模型的validation函数，其中会传入数据集类中的验证集Loader。在DDP环境下，所有的validation函数都会被调用，需要用户自己通过模型的self._rank来进行rank的判断。
8. 若lifeCycle的needSave返回True，则往下进行保存操作，否则跳到10
9. 若lifeCycle的holdThisCheckpoint返回True，则将该次的checkpoint保存到持久通道中，否则则不保存入持久通道。快、慢通道则会根据情况自动进入
10. 若lifeCycle的needContinueTrain返回True，则返回1进行下一个epoch的训练，否则退出训练。

分析进程的主循环流程如下：
1. 如果有预先指定的分析函数，运行并退出（单次分析的情况）
2. 等待命令
3. 得到一个分析脚本路径，加载该脚本中的experience函数作为分析函数
4. 将一个包含了args、model、dataset类的runner作为self对分析函数进行调用，输出会由AnalyzerBuffer同步到主进程
5. 返回2

## 编码指南
使用DLNest进行深度学习开发主要需要修改3个python文件（模型、数据集、生命周期）与config（json形式）文件。  
### root_config.json
该文件作为项目的入口文件，包含了项目所有必须的配置。其默认内容如下：
```json
{
    "save_root":某个目录,
    "model_name":"Model",
    "dataset_name":"Dataset",
    "life_cycle_name":"LifeCycle",
    "checkpoint_args":{
        "max_ckpt_in_slow_track":100,
        "dilation_in_slow_track":100,
        "max_ckpt_in_fast_track":2,
        "max_ckpt_in_consistent_track":1
    },
    "root_file_path":某个目录,
    "model_file_path":"./Model/Model.py",
    "dataset_file_path":"./Dataset/Dataset.py",
    "life_cycle_file_path":"./LifeCycle.py",
    "other_file_paths":[],
    "child_jsons":[
        "./model_config.json",
        "./dataset_config.json"
    ]
}
```
其中save_root指定了保存包的保存位置，root_file_path指定了模型、数据集、生命周期、子json、其他要拷贝文件的默认根目录。这些文件使用相对路径表示时，则相对于root_file_path指定的目录。model_file_path、dataset_file_path、life_cycle_file_path指定了项目的模型、数据集和生命周期代码的路径，而model_name、dataset_name、life_cycle_name则指定了对应文件中模型类、数据集类、生命周期类的类名。other_file_paths指定了除了以上三个文件之外其他需要每次训练需要的文件，比如模型引用的其他代码等。需要注意的是，由于DLNest 0.3采用先拷贝再运行的模式，所以需要所有运行中需要的代码全部计入other_file_paths当中，或在代码中额外指定import搜索路径来手动实现对不太会修改的cuda模块等的引用。在other_file_paths中建议使用相对路径，使用相对路径则会将对应文件保存到保存包中同样的相对路径下，而使用绝对路径则会保存到保存包的根目录。checkpoint_args是与自动保存checkpoint相关的参数。具体的内容在之后模型保存中介绍。child_json可以指定其他config的json文件，最后会由DLNest统一为一个字典输入给各个代码。在不同config发生冲突时，后进入的config会覆盖之前的config中的键值，但当对应键值是一个子字典时，会递归覆盖其中的对应键值，对于其中未出现冲突的键值则不会覆盖。
### model_config与dataset_config
这两个文件分别用来指定模型参数与数据集参数。但实际上，其仅仅是对root_config的补充。最终输入模型与数据集的参数包含了root_config及其所有子json中的所有参数（解决冲突、覆盖之后的结果），在model_config中使用model_config子字典统一储存模型的参数能够更好的整理模型的参数。当然用户也能够采取更加灵活自由的方式进行参数的整理与归类。包括但不限于给root_config添加更多的子config文件、按照模型的结构多层嵌套模型的config，对不同的子模块采用不同的子字典记录参数等
### freq_config.json
在进行训练时，调参是必不可少的过程，但是我们希望频繁的调参能够做到方便且能够只管展示调整的参数。因此设计了freq_config.json来在训练时进行调参。在训练时使用-f选项来指定一个调参json文件，其内部的内容会按照等价于root_config中最后一个子config的模式对默认的参数进行覆盖。
### LifeCycle.py
LifeCycle包含对训练流程的控制，主要需要修改的函数为以下函数：
```python
class LifeCycle(LifeCycleBase):
    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return False

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return False

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(epoch) + " finished!")

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return False

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        return False

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        return False
```
每个函数的用途在对训练/分析进程的介绍中进行了讲解。  
除了这些需要修改的函数以外，在流程的各个阶段，均会有Bxxx（表示Before）函数和Axxx（表示After）函数在各个流程之前和之后调用，若在Bxxx函数中返回字符串Skip，则会跳过这个流程直接执行Axxx。这样的流程包括All, DatasetInit, ModelInit, Train, OneEpoch, ModelOneStep, Visualize, Validation, SaveModel。
### Dataset.py
Dataset包含了数据集的初始化等操作，结构较为简单。其需要实现一个函数init，返回三个变量，分别是一个给模型初始化时使用的dict（用来传递数据给模型），一个训练用Loader（支持enumerate调用）和一个验证用Loader（支持enumerate调用）。数据集的具体实现则根据任务的实际情况进行即可。
### Model.py
Model可能基于ModelBase或ModelBaseTorch。其中ModelBaseTorch作为建议选项支持DDP，自动处理模型参数保存等功能。由于DLNest仅针对Pytorch进行设计，因此这里只对基于ModelBaseTorch的Model进行讲解。要使用ModelBaseTorch只需要如下import即可：
```python
from DLNest.Common.ModelBaseTorch import ModelBaseTorch
```
Model主要需要实现以下几个函数：init, initLog, initOptimizer, runOneStep, visualize, validate。各个函数的作用与注意事项如下：
1. init函数，输入args参数与数据集类给模型的信息datasetInfo，来构建模型。由于Model本身不继承nn.Module，模型的具体实现应该在该类之外进行定义，并在该类中实例化进行引用。实例化之后，需要调用ModelBaseTorch类中已经实现好的self.register()函数进行初始化，方便进行DDP和自动得到保存dict的操作。为了兼容使用DDP，请不要在这个函数中定义优化器。init函数的例子如下：
```python
def init(self,args : dict,datasetInfo : dict = None):
    model = AModel(args)
    self.model = self.register(model,syncBN=True) # SyncBN: 在DDP环境下指定是否使用同步BatchNorm
```  
2. initLog函数返回一个字典，该字典定义了训练过程中需要记录的所有信息，一般包括loss、acc、其他指标等，不包括模型参数。
3. initOptimizer函数用来定义优化器。由于DDP环境下优化器需要在模型全部DDP之后才能定义，因此将这部分进行独立。模型的DDP化会在需要时自动在register函数中被执行，initOptimizer会在init函数之后被调用。
4. runOneStep函数负责进行一步训练。输入包括该step的数据，log（与initLog返回的相同），现在的step编号iter和现在的epoch编号epoch。数据已经被自动处理到了合适的设备上，不用再进行cuda()。可以对tensor使用self._reduceMean或self._reduceSum函数来在DDP环境下reduce所有进程的该张量到rank=0的进程，在非DDP环境下这些函数会直接返回输入的张量。常见的实现如下：
```python
def runOneStep(self,data, log : dict, iter : int, epoch : int):
    self.model.zero_grad()
    x,y = data
    pred = self.model(x)
    loss = self.loss(pred,y)
    loss.backward()
    self.optimizer.step()
    loss = self._reduceMean(loss) 
    log["loss"].append(loss.detach().item())
```
5. visualize函数用来进行可视化，常用的包括使用tensorboard、visdom或直接print来进行
6. validate函数用来进行验证，每一个epoch结束后被调用一次。其输入包括一个验证集loader和log字典。在这里需要注意的是迭代验证集loader时需要自行加载数据到设备。在使用GPU时训练进程已经设置好了默认的cuda设备，直接调用tensor.cuda()即可。可以是用模型的self._envType来得到运行时的环境种类字符串（CPU、GPU、GPUs、DDP），也可以使用模型的self._reduceMean或self._reduceSum函数来跨进程reduce张量到rank=0的进程。一个常见的实现如下：
```python
def validate(self,valLoader,log : dict):
    totalCorrect = 0
    total = 0
    for _iter,data in enumerate(valLoader): # 迭代验证集
        x,y = data
        if self._envType != "CPU": # 若使用GPU，则将数据移到GPU
            x = x.cuda()
            y = y.cuda()
        with torch.no_grad():
            output = self.model(x) # 使用模型进行推理
            _,pred = torch.max(output, 1)
            correct = (pred == y).sum() / y.shape[0] # 得到当前进程的正确率
            correct = self._reduceMean(correct) # 得到所有进程的正确率平均值
            totalCorrect += correct # 统计总计正确率
            total += 1 # 统计step数
    acc = totalCorrect / total # 得到平均正确率
    log["acc"].append(acc.item()) # 记录正确率
```
## Local 命令
使用以下命令能够使用Local服务，该启动方式并不需要启动Server，即前端退出时所有的训练也会退出
```bash
python -m DLNest.Local
```
命令语法如下，其中<>表示描述替代，[]表示可选项，（）表示对参数的描述：

1.      new -d <新建项目的指定路径> [-MNIST （使用则新建一个包含MNIST分类任务代码实现的项目）]；在指定路径新建一个项目
2.      run -c <入口config.json路径> [-f <调参config.json路径>] [-m <估计占用显存（MB），默认为0号显卡总显存的90%>] [-d <对该次训练的描述>] [-ns （使用NOSAVE目录名保存保存包）] [-mc （在单卡显存不满足预估显存时，尝试使用多卡）] [-sd （使用描述字符串作为目录名保存保存包，被ns覆盖）] [-DDP （使用DDP，使用该选项相当于自动设置了mc）] [-CPU （使用CPU进行训练，直接不参与显卡排卡，与DDP互斥）]
3.      continue -r <继续训练的保存包路径> [-c <准备加载的checkpoint编号，默认为最新>] [-d<对该次训练的描述>] [-m <估计占用显存（MB），默认为0号显卡总显存的90%>] [-mc （同run）] [-DDP （同run）] [-CPU （同run）]
4.      analyze -r <分析加载的保存包路径> [-c  <准备加载的checkpoint编号，默认为最新>] [-s <脚本文件搜索目录，默认为项目目录下的AnalyzeScript目录>] [-m （同run）] [-CPU （使用CPU进行分析）]
5.      showTask （打印现有的Task信息）
6.      showDevices （打印可用设备信息）
7.      showDL （打印DLNest各个操作的输出，不包括训练进程输出和分析进程输出）
8.      showAN <想要查看输出的分析任务ID>
9.      runExp <想要运行分析脚本的分析任务ID> <分析脚本名，在分析进程运行时指定的分析脚本目录中查找>
10.         del <想要删除的任务ID>
11.         changeDevices -d <想要使用的设备id，以空格隔离。-1表示CPU（仅做显示，默认除非指定CPU否则不在CPU进行训练或分析）>
12.         exit （安全退出DLNest.Local，也可以使用ctrl + c安全退出）

## 服务器指令
服务器接收http指令，其url、接收参数和作用如下：  
1.      /analyze POST {record_path : str, [script_path : str,checkpoint_ID : int, memory_consumption : int, CPU : (True,False)] }
2.      /change_delay POST {new_delay : int}
3.      /change_devices POST {new_devices_IDs : list[int]}
4.      /change_max_task_per_device POST {new_max : int}
5.      /clear POST {} -> None
6.      /continue_train POST {record_path : str,[checkpoint_ID : int, memory_consumption : int, CPU : (True,False), DDP : (True,False), multi_GPU : (True,False), description : str]}
7.      /del_task POST {task_ID : str} 
8.      /get_analyze_output GET {task_ID : str, styled : (True,False)} -> {offset : int, text : list[styled pair] if styled else str}
9.      /get_devices_info GET {} -> {info : list[dict]}
10.     /get_DLNest_output GET {styled : (True,False)} -> {offset : int, text : list[styled pair] if styled else str }
11.     /get_task_info GET {} -> {info : list[dict]}
12.     /new_proj POST {target_dir : str, [MNIST : (True,False)]}
13.     /run_train POST {config_path : str, [freq_path : str, description : str, memory_consumption : int, CPU : (True,False), DDP : (True,False), multi_GPU : (True,False), description : str, no_save : (True,False), use_description : (True,False)]}
14.     /run_exp POST {task_ID : str, command : str}  

在成功时返回{"status" : "success"}，需要时附加额外信息。出现错误时返回{"status" : "error", "error" : 错误信息}
## Simple 命令
使用以下命令能够使用Simple客户端，该启动方式需要启动Server
```bash
python -m DLNest.Simple
```
命令语法同Local，增加以下
1.      clear （清除DLNest主进程的output缓存）
## Client 命令
命令语法近似于Local，删除以下命令：
1.      showAN
2.      showDL
3.      showDevices
4.      showTask  

增加以下命令：
1.      watch taskID （将某个分析进程加载为当前操作分析进程，能够在Analyzer Output中看到该分析进程的输出）
2.      withdraw （将当前分析进程卸载，不会影响该分析进程的实际运行，但不能够再在Analyzer Output中看到该分析进程的输出）

修改以下命令：
1.      runExp [想要运行分析脚本的分析任务ID] <分析脚本名，在分析进程运行时指定的分析脚本目录中查找>（任务ID可省略，省略时会使用当前加载的分析进程自动填充，若当前未加载分析进程，则runExp操作会在server查找任务时失败输出）
## 单次训练
使用以下命令能够以等效于客户端中run命令的语法进行训练：
```bash
python -m DLNest.Run -c <到root_config的绝对路径> ...
```
输出会同时输出到标准输出与保存包当中
## 单次分析
使用以下命令能够加载保存包运行单个分析脚本：
```bash
python -m DLNest.Analyze -r <到保存包的绝对路径> -s <到分析脚本的绝对路径> ...
```
在python代码中使用以下命令也可在分析器中运行一个函数
```python
from DLNest.Analyze import analyze

def anAnalyzeFunction(self):
    pass

analyze(
    recordPath = <到保存包的绝对路径>,
    expFunc = anAnalyzeFunction
)
```
输出会输出到标准输出当中
## 使用单个模块
DLNest中保存包模块可以单独使用发挥强大的功效。具体可以使用的接口如下：
```python
def __init__(self,configPath : str = "",freqPath : str = ""):
    """
    初始化保存包，参数可以不设置，若不设置参数则需要使用initFromAnExistSavePackage加载，或使用giveArgs指定参数
    若设置则会解析对应的json文件，参数需要至少包含以下内容：
    {
        "save_root" : str,
        "checkpoint_args" :{
            "max_ckpt_in_slow_track" : int,
            "max_ckpt_in_fast_track" : int,
            "max_ckpt_in_consistent_track" : int,
            "dilation_in_slow_track" : int
        }
    }
    """

def giveArgs(self,args : dict):
    """
    给args的字典，必须包含的键值如上
    """

def saveToNewDir(self,overrideSaveName : str = "",copyFiles = True):
    """
    在save_root下新建一个保存包，独立使用时若没有指定model_file_path等路径，需将copyFiles设为False。若overrideSaveName为空，则按调用时间进行保存，否则按给出的SaveName保存
    """

def saveACheckpoint(self,stateDict : dict, holdThisCheckpoint : bool = False):
    """
    将stateDict保存到保存包中，若holdThisCheckpoint为True，则进入持久队列
    """

def saveVisualString(self, visualString : str):
    """
    保存一个文件名为visualString的空文件到保存包的根
    """

def initFromAnExistSavePackage(self,packagePath : str):
    """
    从packagePath加载一个保存包
    """

def getStateDict(self,id = -1,device = "cuda:0"):
    """
    从一个保存包加载一个state_dict到device。若id指定为-1，则加载最新的，若为其他则加载指定id的checkpoint
    """

def setCkptID(self,ckptID : int):
    """
    设置当前的checkpoint ID，比这个id大的都会从保存包的记录中删除，但硬盘中的文件不会被删除。加载非最新checkpoint继续进行训练的请通过此函数设置当前checkpoint
    """
```
## 已知问题
1. 在使用DDP训练，保存模型时退出模型会造成进程泄漏，部分进程不会被关闭
2. Shell Client的输出显示清空时可能会有一行清空不干净
3. 在一些screen内部可能出现client显示异常