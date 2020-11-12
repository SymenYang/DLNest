from prompt_toolkit import Application
from prompt_toolkit.layout.containers import VSplit,HSplit
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from .Windows.CommandInput import CommandInput
from .Windows.ResultsOutput import ResultsOutput,testTask,AnalyzerOutput
from .Windows.TaskInfoShower import TaskInfoShower

from .Communicators.Command import CommandCommunicator
from .Communicators.Output import OutputCommunicator
from .Communicators.Info import InfoCommunicator
import json
from pathlib import Path

class Client:
    def __init__(self,url : str):
        self.url = url
        self.CC = CommandCommunicator(self.url)
        self.DLNestOC = OutputCommunicator(self.url + "/DLNest_buffer")
        self.AnalyzerOC = OutputCommunicator(self.url + "/analyzer_buffer")
        self.InfoC = InfoCommunicator(self.url)
        self.CMDIN = CommandInput(title="DLNest Command Line(F1)",onAccept=self.onCommandAccept)
        self.DLOutput = ResultsOutput(routineTask=self.routineTaskDLO,title = "DLNest Output (F2)",style="class:dlnest_output")
        self.ANOutput = AnalyzerOutput(routineTask=self.routineTaskANO,title = "Analyzer Output (F3)",style="class:analyzer_output",analyzerRoutineTask=self.routineTaskANInfo)
        self.TaskInfo = TaskInfoShower(routineTask = self.routineTaskDLInfo,title = "Tasks (F4)")
        self.w1,self.w2,self.w3,self.w4 = self.CMDIN.getWindow(),self.DLOutput.getWindow(),self.ANOutput.getWindow(),self.TaskInfo.getWindow()
        self.container = HSplit([
            self.w1,
            VSplit([self.w2,self.w3]),
            self.w4
        ])

        self.kb = KeyBindings()
        @self.kb.add('c-c')
        def exit_(event):
            event.app.exit()

        @self.kb.add('f1')
        def focus2(event):
            event.app.layout.focus(self.w1)

        @self.kb.add('f2')
        def focus2(event):
            event.app.layout.focus(self.w2)

        @self.kb.add('f3')
        def focus2(event):
            event.app.layout.focus(self.w3)

        @self.kb.add('f4')
        def focus2(event):
            event.app.layout.focus(self.w4)

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
            "task_info_shower" : "bg:#008bc0"
        })

        self.layout = Layout(self.container,focused_element=self.w1)
        self.app = Application(key_bindings=self.kb, layout=self.layout, full_screen=True,style=self.style)
        self.CC.app = self.app

    def getApp(self):
        return self.app

    def onCommandAccept(self,s : str):
        self.CC.giveACommand(s)

    def routineTaskDLO(self,obj):
        # for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0

        outStyled = self.DLNestOC.getOutput("styled")
        outPlain = self.DLNestOC.getOutput("plain")
        if outStyled is None or outPlain is None:
            obj._count_ = (obj._count_ + 1) % 100
            try:
                obj.lexer.styled_text = [("#ff0000 bold","Connection lossed to " + self.DLNestOC.url + "\n")] 
                obj.shower.text = str(obj._count_) + "\n"
            except Exception as e:
                pass
            return
        obj.lexer.styled_text = [("bold","Connected\n")] + json.loads(outStyled.content)["text"]
        try:
            obj.shower.text = "Connected\n" + json.loads(outPlain.content)["text"]
        except Exception as e:
            pass

    def routineTaskANO(self,obj):
        # for buffer fresh
        if not hasattr(obj,"_count_"):
            obj._count_ = 0

        outStyled = self.AnalyzerOC.getOutput("styled")
        outPlain = self.AnalyzerOC.getOutput("plain")
        if outStyled is None or outPlain is None:
            obj._count_ = (obj._count_ + 1) % 100
            try:
                obj.lexer.styled_text = [("#ff0000 bold","Connection lossed to " + self.AnalyzerOC.url + "\n")] 
                obj.shower.text = str(obj._count_) + "\n"
            except Exception as e:
                pass
            return
        obj.lexer.styled_text = [("bold","Connected\n")] + json.loads(outStyled.content)["text"]
        try:
            obj.shower.text = "Connected\n" + json.loads(outPlain.content)["text"]
        except Exception as e:
            pass

    def routineTaskANInfo(self,obj):
        r = self.InfoC.getAnalyzerTaskInfo()
        if r is None:
            obj.infoText.text = [("","No analyzer task is running")]
            obj.infoWindow.width = 27
            return
        AnInfo = json.loads(r.content)["info"]
        if len(AnInfo) == 0:
            obj.infoText.text = [("","No analyzer task is running")]
            obj.infoWindow.width = 27
        else:
            path = Path(AnInfo["record_path"])
            path_str = path.stem
            text = [
                ("class:analyzer_info_text1","GPU : " + str(AnInfo["GPU_ID"]) + " "),
                ("class:analyzer_info_text2"," CKPT : " + str(AnInfo["checkpoint_ID"]) + " "),
                ("class:analyzer_info_text1"," Path : " + path_str + " ")
            ]
            length = sum([len(item[1]) for item in text])
            obj.infoText.text = text
            obj.infoWindow.width = length

    def routineTaskDLInfo(self,obj):
        r = self.InfoC.getTaskInfo()
        if r is None:
            obj.lexer.taskInfo = []
            obj.shower.text = obj.lexer.get_text()
            return
        TaskInfo = json.loads(r.content)["info"]
        obj.lexer.taskInfo = TaskInfo
        try:
            obj.shower.text = obj.lexer.get_text()
        except Exception as e:
            pass

if __name__ == "__main__":
    client = Client("http://127.0.0.1:9999")
    client.getApp().run()