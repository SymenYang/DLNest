from .Core.DLNestCore import DLNestCore
import json
import tornado.web
import tornado.ioloop
import tornado.options

class DLNestServer:
    def __init__(self):
        tornado.options.define('port',default=9999,type=int,help="DLNest server port")
        tornado.options.define('DLNest-config',default="",type=str,help="DLNest config file")
        tornado.options.parse_command_line()

        self.core = DLNestCore(tornado.options.options.DLNest_config)
        self.app = tornado.web.Application(
            [
                (r'/new_proj',NewProjectHandler,{"core" : self.core}),
                (r'/run_train',RunTrainHandler,{"core" : self.core}),
                (r'/del_task',DelTaskHandler,{"core" : self.core}),
                (r'/sus_task',SusTaskHandler,{"core" : self.core}),
                (r'/reload_task',ReloadTaskHandler,{"core" : self.core}),
                (r'/run_exp',RunExpHandler,{"core" : self.core}),
                (r'/load_model',LoadModelHandler,{"core" : self.core}),
                (r'/release_model',ReleaseModelHandler,{"core" : self.core}),
                (r'/DLNest_buffer',DLNestBufferHandler,{"core" : self.core}),
                (r'/analyzer_buffer',AnalyzerBufferHandler,{"core" : self.core}),
                (r'/task_info',TaskInfoHandler,{"core" : self.core}),
                (r'/analyze_task_info',AnalyzeTaskInfoHandler,{"core" : self.core}),
                (r'/cards_info',CardsInfoHandler,{"core" : self.core}),
                (r'/change_valid_cards',ChangeValidCardsHandler,{"core" : self.core}),
                (r'/change_time_delay',ChangeTimeDelayHandler,{"core" : self.core})
            ]
        )
        self.app.listen(tornado.options.options.port)

    def start(self):
        tornado.ioloop.IOLoop.current().start()

class DLNestHandler(tornado.web.RequestHandler):
    def initialize(self,core : DLNestCore):
        self.core = core

class NewProjectHandler(DLNestHandler):
    def post(self):
        targetDir = None
        try:
            targetDir = self.get_argument("target_dir")
            self.core.newProj(targetDir)
            self.write({
                "status" : "success"
            })
            print("New Project : ", targetDir)
        except Exception as e:
            if targetDir is None:
                self.write({
                    "status" : "error",
                    "error" : "No target_dir is given."
                })
            else:
                raise(e)

class RunTrainHandler(DLNestHandler):
    def post(self):
        try:
            rootConfig = self.get_argument("root_config")
            description = self.get_argument("description",default = "")
            freqConfig = self.get_argument("freq_config",default = "")
            memoryConsumption = int(self.get_argument("memory_consumption",default = -1))
            jumpInLine = True if self.get_argument("jump_in_line",default = "False") == "True" else False
            multiCard = True if self.get_argument("multi_card",default = "False") == "True" else False
            noSave = True if self.get_argument("no_save",default = "False") == "True" else False
            self.core.runTrain(
                rootConfig=rootConfig,
                description = description,
                freqConfig = freqConfig,
                memoryConsumption = memoryConsumption,
                jumpInLine = jumpInLine,
                multiCard = multiCard,
                noSave = noSave
            )
            self.write({
                "status" : "success"
            })
            print("RunTrain",rootConfig)
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : "No root_config is given."
            })

class DelTaskHandler(DLNestHandler):
    def post(self):
        try:
            taskID = self.get_argument("task_ID")
            self.core.delTask(taskID)
            self.write({
                "status" : "success"
            })
            print("Del Task", taskID)
        except Exception as e:
            print(e)
            self.write({
                "status" : "error",
                "error" : "No task_ID is given."
            })

class SusTaskHandler(DLNestHandler):
    def post(self):
        try:
            taskID = self.get_argument("task_ID")
            self.core.susTask(taskID)
            self.write({
                "status" : "success"
            })
            print("Suspend Task", taskID)
        except Exception as e:
            print(e)
            self.write({
                "status" : "error",
                "error" : "No task_ID is given."
            })

class ReloadTaskHandler(DLNestHandler):
    def post(self):
        try:
            taskID = self.get_argument("task_ID")
            self.core.reloadTask(taskID)
            self.write({
                "status" : "success"
            })
            print("Reload Task", taskID)
        except Exception as e:
            print(e)
            self.write({
                "status" : "error",
                "error" : "No task_ID is given."
            })

class RunExpHandler(DLNestHandler):
    def post(self):
        try:
            command = self.get_argument("command")
            self.core.runExp(command)
            print("Run exp ", command)
            self.write({
                "status" : "success"
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : "No command is given."
            })

class LoadModelHandler(DLNestHandler):
    def post(self):
        try:
            recordPath = self.get_argument("record_path")
            scriptPath = self.get_argument("script_path")
            checkpointID = int(self.get_argument("checkpoint_ID"))
            memoryConsumption = int(self.get_argument("memory_consumption",default = -1))
            self.core.loadModel(
                recordPath = recordPath,
                scriptPath = scriptPath,
                checkpointID = checkpointID,
                memoryConsumption = memoryConsumption
            )
            self.write({
                "status" : "success"
            })
            print("Load Model", recordPath)
        except Exception as e:
            print(e)
            self.write({
                "status" : "error",
                "error" : "No record_path or script_path or checkpoint_ID is given."
            })

class ReleaseModelHandler(DLNestHandler):
    def get(self):
        self.core.releaseModel()
        self.write({
            "status" : "success"
        })
        print("Release Model")

class DLNestBufferHandler(DLNestHandler):
    def get(self):
        textType = self.get_query_argument("type",default="plain")
        if textType == "styled":
            ret = self.core.getDLNestStyledOutput()
        else:
            ret = self.core.getDLNestOutput()
        self.write({
            "offset" : ret[0],
            "text" : ret[1]
        })

class AnalyzerBufferHandler(DLNestHandler):
    def get(self):
        textType = self.get_query_argument("type",default="plain")
        if textType == "styled":
            ret = self.core.getAnalyzerStyledOutput()
        else:
            ret = self.core.getAnalyzerOutput()
        self.write({
            "offset" : ret[0],
            "text" : ret[1]
        })

class TaskInfoHandler(DLNestHandler):
    def get(self):
        info = self.core.getTasks()
        self.write({
            "info" : info
        })

class AnalyzeTaskInfoHandler(DLNestHandler):
    def get(self):
        info = self.core.getAnalyzeTask()
        self.write({
            "info" : info
        })

class CardsInfoHandler(DLNestHandler):
    def get(self):
        info = self.core.getCardsInfo()
        self.write({
            "info" : info
        })

class ChangeValidCardsHandler(DLNestHandler):
    def post(self):
        try:
            newCards = self.get_arguments("cards")
            newCards = [int(item) for item in newCards]
            self.core.changeCards(newCards)
            self.write({
                "status" : "success"
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })

class ChangeTimeDelayHandler(DLNestHandler):
    def post(self):
        try: 
            newDelay = self.get_argument("delay")
            self.core.changeTimeDelay(int(newDelay))
            self.write({
                "status" : "success"
            })
        except Exception as e:
            self.write({
                "status" : "error",
                "error" : str(e)
            })

if __name__ == "__main__":
    server = DLNestServer()
    server.start()