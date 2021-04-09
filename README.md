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
待更新
## 编码指南
使用DLNest进行深度学习开发主要需要修改3个python文件与三个config.json文件。具体细节待更新
## Local 命令
使用以下命令能够使用Local服务，该启动方式并不需要启动Server，即前端退出时所有的训练也会退出
```bash
python -m DLNest.Local
```
命令语法待更新
## 服务器指令
待更新
## Simple 命令
使用以下命令能够使用Simple客户端，该启动方式需要启动Server
```bash
python -m DLNest.Simple
```
命令语法待更新
## Client 命令
命令语法待更新
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
## 已知问题
1. 在使用DDP训练，保存模型时退出模型会造成进程泄漏，部分进程不会被关闭
2. Shell Client的输出显示清空时可能会有一行清空不干净