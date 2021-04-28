from prompt_toolkit import Application
from prompt_toolkit.layout.containers import VSplit,HSplit
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

import json
from pathlib import Path
import argparse
import os

from DLNest.ShellClient.Windows.DevicesInfoShower import DevicesInfoShower
from DLNest.ShellClient.Windows.CommandInput import CommandInput
from DLNest.ShellClient.Windows.ResultsOutput import ResultsOutput,AnalyzeOutput
from DLNest.ShellClient.Windows.TaskInfoShower import TaskInfoShower
from DLNest.ShellClient.Communicator import Communicator

class Client:
    def __init__(self, url : str = "127.0.0.1", port : int = "9999"):
        self.communicator = Communicator(url,port)

        self.CMDIN = CommandInput(title="DLNest Command Line(F1)",onAccept=self.onCommandAccept)
        self.w1 = self.CMDIN.getWindow()

        self.DLOutput = ResultsOutput(routineTask=self.routineTaskDLOutput,title = "DLNest Output (F2)",style="class:dlnest_output")
        self.w2 = self.DLOutput.getWindow()

        self.ANOutput = AnalyzeOutput(routineTask=self.routineTaskANOutput,title = "Analyzer Output (F3)",style="class:analyzer_output")
        self.w3 = self.ANOutput.getWindow()
        self.analyzeTaskID = ""

        self.TaskInfo = TaskInfoShower(routineTask = self.routineTaskInfo,title = "Tasks (F4)")
        self.w4 = self.TaskInfo.getWindow()

        self.DevicesInfo = DevicesInfoShower(routineTask = self.routineTaskDevices, title = "Devices (F5)")
        self.w5 = self.DevicesInfo.getWindow()

        self.container_fat = HSplit([
            self.w1,
            VSplit([self.w2,self.w3]),
            VSplit([self.w4,self.w5])
        ])
        self.container_tall = HSplit([
            self.w1,
            self.w2,
            self.w3,
            self.w4,
            self.w5
        ])
        
        self.kb = KeyBindings()
        @self.kb.add('c-c')
        def exit_(event):
            event.app.exit()

        @self.kb.add('f1')
        def focus1(event):
            event.app.layout.focus(self.w1)

        @self.kb.add('f2')
        def focus2(event):
            event.app.layout.focus(self.w2)

        @self.kb.add('f3')
        def focus3(event):
            event.app.layout.focus(self.w3)

        @self.kb.add('f4')
        def focus4(event):
            event.app.layout.focus(self.w4)

        @self.kb.add('f5')
        def focus5(event):
            event.app.layout.focus(self.w5)
        
        self.style = Style.from_dict({
            "frame.border" : "fg:#ffb6c1",
            "frame.title" : "fg:#1ef0ff",
            "command_frame" : "bg:#008b8b",
            "dlnest_output" : "bg:#451a4a",
            "analyzer_output" : "bg:#451a4a",
            "analyzer_info_label" : "bg:#da70d6",
            "analyzer_info_text1" : "bg:#3f3f00",
            "analyzer_info_text2" : "bg:#ff00ff",
            "running_task_status" : "bg:#a01010 bold",
            "running_task_id" : "bg:#303030",
            "running_task_gpu" : "bg:#556b2f",
            "running_task_des" : "bg:#c71585",
            "running_task_time" : "bg:#2e3b37",
            "pending_task_status" : "bg:#1010a0 bold",
            "pending_task_id" : "bg:#303030",
            "pending_task_gpu" : "bg:#556b2f",
            "pending_task_des" : "bg:#c71585",
            "pending_task_time" : "bg:#2e3b37",
            "suspend_task_status" : "bg:#10a010 bold",
            "suspend_task_id" : "bg:#303030",
            "suspend_task_gpu" : "bg:#556b2f",
            "suspend_task_des" : "bg:#c71585",
            "suspend_task_time" : "bg:#2e3b37",
            "task_info_shower" : "bg:#008bc0",
            "devices_info_shower" : "bg:#008bc0",
            "devices_id" : "bg:#303030",
            "devices_status_valid" : "bg:#3cb371 bold",
            "devices_status_break" : "bg:#a01010 bold",
            "devices_free_memory" : "bg:#556b2f",
            "devices_tasks" :  "bg:#c71585"
        })

        self.layout = Layout(self.container_fat,focused_element=self.w1)
        self.app = Application(key_bindings=self.kb, layout=self.layout, full_screen=True,style=self.style)
        self.app._on_resize = self.on_resize

    def on_resize(self):
        cols, rows = os.get_terminal_size(0)
        focused_element = self.layout.current_window
        if cols >= 2 * rows: # fat
            self.app.layout = Layout(self.container_fat,focused_element=focused_element)
        else: # tall
            self.app.layout = Layout(self.container_tall,focused_element=focused_element)
        
        self.app.renderer.erase(leave_alternate_screen=False)
        self.app._request_absolute_cursor_position()
        self.app._redraw()

    def getApp(self):
        return self.app

    def onCommandAccept(self,s : str):
        commandWordList = s.split(" ")
        while "" in commandWordList:
            commandWordList.remove("")

        # no command
        if len(commandWordList) <= 0:
            return

        if commandWordList[0] == "watch":
            self.analyzeTaskID = commandWordList[1]
        elif commandWordList[0] == "withdraw":
            self.analyzeTaskID = ""
        
        if commandWordList[0] == "runExp":
            if len(commandWordList) != 3:
                if self.analyzeTaskID != "":
                    commandWordList = [commandWordList[0],self.analyzeTaskID,commandWordList[1]]
                else:
                    return

        ret = self.communicator.giveACommand(commandWordList)

        if commandWordList[0] == "del":
            if ret["status"] == "success" and commandWordList[1] == self.analyzeTaskID:
                self.analyzeTaskID = ""

        if "exit" in ret:
            self.app.exit()

    def routineTaskDLOutput(self, obj):
        #for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0

        outStyledDict = self.communicator.giveACommand(["showDL","-s"])
        outPlainDict = self.communicator.giveACommand(["showDL"])
        if "text" in outStyledDict and "text" in outPlainDict:
            try:
                obj.lexer.styled_text = outStyledDict["text"]
                obj.shower.text = outPlainDict["text"]
            except Exception as e:
                pass

    def routineTaskANOutput(self, obj):
        #for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0
        
        if self.analyzeTaskID == "":
            obj.lexer.styled_text = []
            obj.shower.text = ""
            obj.infoText.text = [("","No valid analyzer task is running")]
            obj.infoWindow.width = 33
            return

        outStyledDict = self.communicator.giveACommand(["showAN","-t",self.analyzeTaskID,"-s"])
        outPlainDict = self.communicator.giveACommand(["showAN","-t",self.analyzeTaskID])
        if "text" in outStyledDict and "text" in outPlainDict:
            try:
                obj.lexer.styled_text = outStyledDict["text"]
                obj.shower.text = outPlainDict["text"]
                obj.infoText.text = [("class:analyzer_info_text1",self.analyzeTaskID)]
                obj.infoWindow.width = len(self.analyzeTaskID)
            except Exception as e:
                pass
        else:
            self.analyzeTaskID = ""

    def routineTaskInfo(self,obj):
        # for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0
        
        r = self.communicator.giveACommand(["showTask"])
        if r["status"] != "success":
            obj.lexer.taskInfo = []
            obj.shower.text = obj.lexer.get_text()
            return
        taskInfo = r["info"]
        try:
            obj.lexer.taskInfo = taskInfo
            obj.shower.text = obj.lexer.get_text()
        except Exception as e:
            pass

    def routineTaskDevices(self, obj):
        # for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0

        r = self.communicator.giveACommand(["showDevice"])
        if r["status"] != "success":
            obj.lexer.devicesInfo = []
            obj.shower.text = obj.lexer.get_text()
            return
        obj.lexer.devicesInfo = r["info"]
        try:
            obj.shower.text = obj.lexer.get_text()
        except Exception as e:
            pass

def startClient():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u",type=str, default="127.0.0.1",help="DLNest server address")
    parser.add_argument("-p",type=int, default=9999, help = "DLNest server port")
    args=parser.parse_args()
    client = Client(url = args.u, port = args.p)
    client.getApp().run()