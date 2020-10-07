# DLNest
DLNest是一个深度学习训练\实验框架，轻依赖于pytorch，实现单机自动排卡，训练任务列表，高可复现自动记录，自动载入模型进行实验等功能
## 安装
依赖
```
python >= 3.6.9
pytorch >= 1.3.0
APScheduler >= 3.6.3
nvidia-ml-py >= 375.53.1
prompt-toolkit >= 2.0.10
```
运行DLNest:
```
python DLNest.py [-c]
```
使用-c指定DLNest的配置文件，默认为DLNest_config.json。该配置文件可以设置DLNest可见的显卡(default : 全部)，默认任务启动时长(default : 60s)，单卡最多任务数(default : 无限制)。见DLNest_config.json
## 使用
### 1. 新建基于DLNest的项目
在DLNest中，键入
```
new -d <希望新建项目的绝对路径>
```
在新建的路径内，目录结构如下：
```python
｜--<Custom Project Dir>
    |--AnalyzeScripts # 建议用于存放在已训练好模型上进行的实验代码
    |  |--test.py # 示例实验代码
    |--Dataset # 存放数据集代码文件
    |  |--DatasetBase.py # DLNest使用的数据集类的基类
    |--Model # 存放模型代码文件
    |  |--ModelBase.py # DLNest使用的模型类的基类
    |--Saves # 用以存放训练输出、checkpoint、参数等
    |--dataset_config.json # 数据集参数
    |--model_config.json # 模型参数
    |--freq_config.json # 训练时频繁修改的参数
    |--root_config.json # 根参数，或称任务参数
    |--LifeCycleBase.py # 生命周期钩子代码
```
### 2. 修改配置文件
基于DLNest的项目，存在一个主配置文件，即`root_config.json`，主要需要修改的有
```
model_name : 在模型文件中，模型类的名称，如ResNet
dataset_name : 数据集类的名称，如ImageNetDataset
life_cycle_name : 生命周期类的名称，如MyLifeCycle
model_file_path,dataset_file_path,life_cycle_file_path : 运行时模型、数据集、生命周期文件的绝对路径（建议在freq_config中设置，将会覆盖root_config中的同名设置
```
`root_config.json`中有`child_jsons`键，DLNest将会递归各个配置文件json，并按照dfs序，后出现的覆盖先出现的配置，最终使用freq_config中的配置覆盖之前的所有配置文件的配置，再进行训练。`child_jsons`键中能够使用相对于`root_config.json`的相对路径。覆盖的过程中，会对dict类型的键进行递归覆盖，见如下的例子
```
root_config.json:
{
    "example":{
        "a" : "a",
        "b" : "b",
    }
}
freq_config.json:
{
    "example":{
        "a" : "c"
    }
}
DLNest处理后的args字典:
{
    "example":{
        "a" : "c",
        "b" : "b",
    }
}
```
默认在`root_config.json`中链接了`dataset_config.json`和`model_config.json`作为子json，用于存放数据集与模型的参数。因此，可在`freq_config.json`中增量修改模型或数据集参数，不会完全覆盖所有参数。

** 在数据集类和模型中仍然能够访问到所有的参数，模型和数据集代码需要自己处理已经变化为字典的参数

### 3. 编写Dataset代码
继承DatasetBase类编写Dataset代码，需要修改`afterInit()`函数，返回如下信息
```
传递给模型的信息字典，训练DataLoader，验证DataLoader
```

### 4. 编写Model代码
继承ModelBase类编写Model代码，其中各个函数的作用如下：
```python
def __init__(self,args : dict,datasetInfo : dict = None):
    # 按照args和数据集传递的信息datasetInfo初始化模型
    
def initLog(self):
    # 返回一个字典，表示训练中需要保存的log，例如损失函数数组，验证指标数组等

def getSaveDict(self):
    # 返回一个字典，用于保存训练状态，如模型参数，损失函数、验证指标等

def loadSaveDict(self,saveDict : dict):
    # 接收一个字典，并恢复训练状态，在getSaveDict返回的基础上会有一个新增的键"epoch"表示已经运行过的epoch数

def runOneStep(self,data,log : dict,iter : int,epoch : int):
    # 运行一步的训练，输入包括数据集给出的数据data,log，当前在epoch中的step数，已经当前的epoch数

def visualize(self):
    # 进行可视化的代码，阻塞进行，不建议运行时间过长

def validate(self,valLoader,log : dict):
    # 进行验证的代码，能够拿到验证集的DataLoader和log
```

### 5. 编写LifeCycle代码
继承LifeCycleBase类编写LifeCycle代码，主要需要修改的函数如下：
```python
def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
    # 指示DLNest是否需要进行可视化，默认为False（在每个step后调用）

def needSaveModel(self, epoch : int, logdict : dict, args : dict):
    # 指示DLNest是否需要保存模型，默认为True（在每个epoch后调用）

def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
    # 指示DLNest是否需要将模型保存到Consistent Track中，默认为False（在每个epoch后调用）

def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
    # 指示DLNest是否需要继续训练，默认为False即不继续训练（在每个epoch后调用）
```

### 6. 修改`freq_config.json`
将常修改的，需要覆盖默认参数的参数写入`freq_config.json`，常用的包括模型文件的路径，调试过程的模型参数等等

### 7. 运行训练
在运行着的DLNest交互式命令行中输入如下命令:
```shell
run -c <到root_config.json的绝对路径> [-f] <到freq_config.json的绝对路径> [-m] <估计的显存占用(MB)，默认为显卡满显存的90%> [-j] <若为True则插入等待队列的最前端> [-d] <一个描述这次训练的字符串>
```

### 8.查看输出
DLNest会在项目的Saves文件夹内按照开始运行的时间生成对每次训练生成一个文件夹，其中主要包含了：
```python
|--2020-09-10_12:00:00_0 #开始运行的时间
   |--Checkpoints
   |  |--epoch_<num>.ckpt # 第num个epoch的保存文件
   |--_Model.py # 存档的模型代码
   |--_Dataset.py # 存档的数据集代码
   |--_LifeCycle.py # 存档的生命周期代码
   |--_output.txt # 重定向后的模型输出
   |--args.json # 参数json
   |--other files # 手动在other_file_paths中指定的文件，主要用于辅助Model，Dataset等的运行（例如它们import的代码）
```
在args.json中，有键`_description`和`_pid`,分别保存了训练时的描述和训练进程的pid，可以用来提前终止该训练进程

### 9.运行实验代码
训练完成后，DLNest可以加载模型并动态的运行实验代码  
在DLNest中运行如下命令启动一个实验
```
analyze -r <某次训练的保存文件夹的绝对路径，如/root/proj/Saves/2020-09-10_12:00:00_0> -s <运行实验的脚本文件目录> -c <想要载入的epoch编号> [-m] <估计的显存占用(MB)，默认为显卡满显存的90%>
```
在实验脚本文件目录中，新建实验脚本，例如默认给出的test.py：
```python
def experience(self):
    print(self.args,self.model,self.dataset)
```
每个实验脚本文件需要包含一个experience函数，并有且只有一个需要传值的参数。传入为一个拥有训练时所有参数，加载好checkpoint的模型和完成初始化的数据集的简单类。可以在函数中自由访问这些变量  
这个命令会启动DLNest Analyzer交互式命令行，通过输入如下命令运行实验代码
```
run <实验脚本名，如test>
```
DLNest Analyzer会自动寻找对应名称的py文件，加载其中的experience函数并运行。不用担心experience函数写的有问题，只要不exit(0)，DLNest Analyzer进程大概率不会退出，模型也不需要重新加载  
值得一提的是，实验脚本文件可以在DLNest Analyzer运行的过程中动态修改、新建，可以方便、灵活的进行实验，且不用重新加载模型  
在DLNest Analyzer中键入exit命令即可退出DLNest Analyzer  
** 虽然DLNest Analyzer阻塞了DLNest进程的交互，但DLNest Analyzer与DLNest并不是同一个进程，在DLNest Analyzer中停留不影响DLNest中的实验和计划实验的运行（但可能会通过占用GPU资源影响到）
## 修改计划
- [ ] 任务请求化（http）
- [ ] 复制文件夹 
- [ ] 支持相对路径
- [ ] 多卡支持