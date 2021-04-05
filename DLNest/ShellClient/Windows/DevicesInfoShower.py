from apscheduler.schedulers.background import BackgroundScheduler
from typing import Callable, Iterable, List, Optional

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

class DevicesLexer(Lexer):
    def __init__(self):
        self.devicesInfo = []
        self.IDLength = 5
        self.freeMemoryLength = 10
        self.runningTasksLength = 3
    
    def get_text(self):
        return "".join(["\n" for _ in self.devicesInfo])

    def lex_document(self,document : Document) -> Callable[[int], StyleAndTextTuples]:
        def get_line(lineno : int) -> StyleAndTextTuples:
            try:
                device = self.devicesInfo[lineno]
                ID = "GPU " + str(device["ID"]) if device["ID"] != -1 else "CPU "
                isBreak = "Break" if device["is_break"] else "Valid"
                break_class = "break" if device["is_break"] else "valid"
                freeMemory = str(int(device["free_memory"])) + " MB"
                runningTasks = str(len(device["running_tasks"]))

                if len(ID) < self.IDLength:
                    ID = " " * (self.IDLength - len(ID)) + ID

                if len(freeMemory) < self.freeMemoryLength:
                    freeMemory = " " * (self.freeMemoryLength - len(freeMemory)) + freeMemory
                
                if len(runningTasks) < self.runningTasksLength:
                    runningTasks = " " * (self.runningTasksLength - len(runningTasks)) + runningTasks

                return [
                    ("class:devices_id"                   ,"Devices: " + ID + " "),
                    ("class:devices_status_" + break_class," " + isBreak + " "),
                    ("class:devices_free_memory"          ," F-Memory: " + freeMemory + " "),
                    ("class:devices_tasks"                ," #Tasks: " + runningTasks + " ")
                ]
            except Exception as e:
                return []
        
        return get_line

class DevicesInfoShower:
    def __init__(
        self,
        title : str = "Tasks",
        routineTask = None,
        freq : int = 1,
        style : str = "class:devices_info_shower"
    ):
        self.title = title
        self.routineTask = routineTask
        self.scheduler = BackgroundScheduler()
        if not routineTask is None:
            self.scheduler.add_job(self.routineTask,'interval',seconds=freq,args=[self])
        self.scheduler.start()
        self.lexer = DevicesLexer()

        self.shower = SCTextArea(
            lexer = self.lexer,
            wrap_lines=False
        )

        self.style = style
        self.window = Frame(
            self.shower,
            self.title,
            style = self.style,
            height=10,
            width=60
        )

    def getWindow(self):
        return self.window