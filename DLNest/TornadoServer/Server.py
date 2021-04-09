import tornado.web
import tornado.ioloop
import tornado.options

from DLNest.Operations.Analyze import analyze
from DLNest.Operations.ChangeDelay import changeDelay
from DLNest.Operations.ChangeDevices import changeDevices
from DLNest.Operations.ChangeMaxTaskPerDevice import changeMaxTaskPerDevice
from DLNest.Operations.ClearDLNestOutput import clearDLNestOutput
from DLNest.Operations.ContinueTrain import continueTrain
from DLNest.Operations.DelATask import delATask
from DLNest.Operations.GetAnalyzeOutput import getAnalyzeOutput
from DLNest.Operations.GetDevicesInformation import getDevicesInformation
from DLNest.Operations.GetDLNestOutput import getDLNestOutput
from DLNest.Operations.GetTasksInformation import getTasksInformation
from DLNest.Operations.New import new
from DLNest.Operations.Run import run
from DLNest.Operations.RunExp import runExp
from DLNest.Operations.SafeExit import safeExit

from DLNest.Output.DLNestBuffer import DLNestBuffer

import sys
import traceback
from threading import Lock

class DLNestServer:
    def __init__(self):
        tornado.options.define("port", default = 9999, type = int, help = "DLNest server port")
        tornado.options.parse_command_line()

        self.outputBuffer = DLNestBuffer()
        self.singleLock = Lock()
        self.app = tornado.web.Application(
            [
                (r'/analyze',AnalyzeHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/change_delay',ChangeDelayHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/change_devices',ChangeDevicesHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/change_max_task_per_device',ChangeMaxTaskPerDeviceHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/clear',ClearHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/continue_train',ContinueTrainHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/del_task',DelTaskHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/get_analyze_output',GetAnalyzeOutputHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/get_devices_info',GetDevicesInfoHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/get_DLNest_output',GetDLNestOutputHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/get_task_info',GetTaskInfoHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/new_proj',NewProjectHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/run_train',RunTrainHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock}),
                (r'/run_exp',RunExpHandler,{"outputBuffer" : self.outputBuffer, "lock" : self.singleLock})
            ]
        )
        self.app.listen(tornado.options.options.port)
    
    def start(self):
        tornado.ioloop.IOLoop.current().start()


class DLNestHandler(tornado.web.RequestHandler):
    def initialize(self,outputBuffer : DLNestBuffer, lock : Lock):
        self.lock = lock
        self.outputBuffer = outputBuffer
        self.DEBUG = False
    
    def beforeExec(self,appName : str = "DLNest"):
        if self.lock.acquire():
            self.stdout = sys.stdout
            self.stderr = sys.stderr
            sys.stdout = self.outputBuffer
            sys.stderr = self.outputBuffer
            self.outputBuffer.appName = appName

    def afterExec(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.outputBuffer.appName = "DLNest Server"
        self.lock.release()


class AnalyzeHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Analyze")
            recordPath = self.get_argument("record_path")
            scriptPath = self.get_argument("script_path", default = "")
            checkpointID = int(self.get_argument("checkpoint_ID", default = "-1"))
            memoryConsumption = int(self.get_argument("memory_consumption", default = "-1"))
            CPU = True if self.get_argument("CPU",default = "False") == "True" else False
            analyze(
                recordPath = recordPath,
                scriptPath = scriptPath,
                checkpointID = checkpointID,
                CPU = CPU,
                memoryConsumption = memoryConsumption,
                otherArgs = {}
            )
            self.write({
                "status" : "success"
            })
            print("Analyze",recordPath)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class ChangeDelayHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Change Delay")
            newDelay = int(self.get_argument("new_delay"))
            
            changeDelay(newDelay)
            
            self.write({
                "status" : "success"
            })
            print("Change delay",newDelay)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class ChangeDevicesHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Change devices")
            newDevicesStr = self.get_arguments("new_devices_IDs")
            newDevices = [int(item) for item in newDevicesStr]

            changeDevices(newDevices)
            
            self.write({
                "status" : "success"
            })
            print("Change devices",",".join(newDevicesStr))
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class ChangeMaxTaskPerDeviceHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Change max tasks per device")
            newMax = int(self.get_argument("new_max"))

            changeMaxTaskPerDevice(newMax)
            
            self.write({
                "status" : "success"
            })
            print("Change max tasks per device to",newMax)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class ClearHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Clear")

            clearDLNestOutput()

            self.write({
                "status" : "success"
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class ContinueTrainHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Continue train")
            
            recordPath = self.get_argument("record_path")
            checkpointID = int(self.get_argument("checkpoint_ID", default = "-1"))
            memoryConsumption = int(self.get_argument("memory_consumption", default = "-1"))
            CPU = True if self.get_argument("CPU",default = "False") == "True" else False
            DDP = True if self.get_argument("DDP",default = "False") == "True" else False
            multiGPU = True if self.get_argument("multi_GPU", default = "False") == "True" else False
            description = self.get_argument("description", default = "")
            
            continueTrain(
                recordPath = recordPath,
                checkpointID = checkpointID,
                memoryConsumption = memoryConsumption,
                CPU = CPU,
                DDP = DDP,
                multiGPU = multiGPU,
                description = description,
                otherArgs = {}
            )

            self.write({
                "status" : "success"
            })
            print("Continue train",recordPath)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class DelTaskHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Del task")

            taskID = self.get_argument("task_ID")

            delATask(taskID)

            self.write({
                "status" : "success"
            })
            print("Deleted a task", taskID)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class GetAnalyzeOutputHandler(DLNestHandler):
    def get(self):
        try:
            self.beforeExec("Get analyze output")

            taskID = self.get_argument("task_ID")
            styled = True if self.get_argument("styled",default = True) == "True" else False

            data = getAnalyzeOutput(taskID = taskID, style = styled)

            self.write({
                "status" : "success",
                "offset" : data[0],
                "text" : data[1]
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class GetDevicesInfoHandler(DLNestHandler):
    def get(self):
        try:
            self.beforeExec("Get devices information")

            data = getDevicesInformation()

            self.write({
                "status" : "success",
                "info" : data
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class GetDLNestOutputHandler(DLNestHandler):
    def get(self):
        try:
            self.beforeExec("Get DLNest output")

            styled = True if self.get_argument("styled",default = True) == "True" else False

            data = getDLNestOutput(style = styled)

            self.write({
                "status" : "success",
                "offset" : data[0],
                "text" : data[1]
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class GetTaskInfoHandler(DLNestHandler):
    def get(self):
        try:
            self.beforeExec("Get task information")

            data = getTasksInformation()

            self.write({
                "status" : "success",
                "info" : data
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class NewProjectHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("New project")

            targetDir = self.get_argument("target_dir")
            MNIST = True if self.get_argument("MNIST",default = "False") == "True" else False
            
            new(targetDir,MNIST = MNIST)

            self.write({
                "status" : "success"
            })
            print("New project in",targetDir)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class RunTrainHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Run train")

            configPath = self.get_argument("config_path")
            freqPath = self.get_argument("freq_path")
            description = self.get_argument("description", default = "")
            memoryConsumption = int(self.get_argument("memory_consumption", default = "-1"))
            CPU = True if self.get_argument("CPU",default = "False") == "True" else False
            DDP = True if self.get_argument("DDP",default = "False") == "True" else False
            multiGPU = True if self.get_argument("multi_GPU", default = "False") == "True" else False
            noSave = True if self.get_argument("no_save",default = "False") == "True" else False
            useDescriptionToSave = True if self.get_argument("use_description", default = "False") == "True" else False

            run(
                configPath = configPath,
                freqPath = freqPath,
                description = description,
                memoryConsumption = memoryConsumption,
                CPU = CPU,
                DDP = DDP,
                multiGPU = multiGPU,
                noSave = noSave,
                useDescriptionToSave = useDescriptionToSave,
                otherArgs = {}
            )

            self.write({
                "status" : "success"
            })
            print("Run train from ",configPath)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])


class RunExpHandler(DLNestHandler):
    def post(self):
        try:
            self.beforeExec("Run experiment")

            taskID = self.get_argument("task_ID")
            command = self.get_argument("command")
            runExp(taskID = taskID, command = command)

            self.write({
                "status" : "success"
            })
            print("Run exp",command,"to",taskID)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })
            traceback.print_exc()
        finally:
            self.afterExec()
            if self.DEBUG:
                print(self.outputBuffer.getPlainText()[1])