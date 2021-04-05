from apscheduler.schedulers.background import BackgroundScheduler
from typing import Callable, Iterable, List, Optional
from pathlib import Path

from prompt_toolkit.widgets import Frame,TextArea,Label,Box
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import VSplit, Window,HSplit
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text.base import StyleAndTextTuples

from DLNest.ShellClient.Windows.SCTextArea import SCTextArea

class taskLexer(Lexer):
    def __init__(self):
        self.taskInfo = []
        self.saveLength = 24
        self.devicesLength = 10

    def get_text(self):
        return "".join(["\n" for _ in self.taskInfo])

    def lex_document(self,document : Document) -> Callable[[int], StyleAndTextTuples]:
        def get_line(lineno : int) -> StyleAndTextTuples:
            try:
                task = self.taskInfo[lineno]
                style_base = "class:pending_task"
                if task["status"] == "Running":
                    style_base = "class:running_task"
                # elif task["status"] == "Suspend":
                #     style_base = "class:suspend_task"
                ID = task["ID"]
                taskType = " Train " if task["type"] == "Train" else "Analyze"
                devices = " ".join([str(item) for item in task["devices"]])
                description = task["description"] if "description" in task else ""
                save = Path(task["args"]["root_file_path"]).stem
                
                if len(save) < self.saveLength:
                    save += " " * (self.saveLength - len(save))
                elif len(save) > self.saveLength:
                    save = save[:self.saveLength - 2] + ".."
                
                if len(devices) < self.devicesLength:
                    devices += " " * (self.devicesLength - len(devices))

                return [
                    (style_base + "_status" , "Status : " + task["status"] + " "),
                    (style_base + "_type", " Type : " + taskType + " "),
                    (style_base +"_id" , " ID : " + ID + " "),
                    (style_base +"_device" , " Devices : " + devices + " "),
                    (style_base +"_time" , " Folder : " + save + " "),
                    (style_base +"_des" , " Note : " + description + " ")
                ]
            except Exception as e:
                return []
        
        return get_line

class TaskInfoShower:
    def __init__(
        self, 
        title : str = "Tasks",
        routineTask = None,
        freq : int = 1,
        style : str = "class:task_info_shower"
    ):

        self.title = title
        self.routineTask = routineTask
        self.scheduler = BackgroundScheduler()
        if not routineTask is None:
            self.scheduler.add_job(self.routineTask,'interval',seconds=freq,args=[self])
        self.scheduler.start()
        self.lexer =taskLexer()

        self.shower = SCTextArea(
            lexer = self.lexer,
            wrap_lines=False
        )

        self.style = style
        self.window = Frame(
            self.shower,
            self.title,
            style = self.style,
            height=10
        )
    
    def getWindow(self):
        return self.window