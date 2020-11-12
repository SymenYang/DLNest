from prompt_toolkit.widgets import Frame,TextArea,Label,Box
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from typing import Callable, Iterable, List, Optional
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import VSplit, Window,HSplit
from apscheduler.schedulers.background import BackgroundScheduler
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text.base import StyleAndTextTuples

from .ResultsOutput import MyTextArea

class taskLexer(Lexer):
    def __init__(self):
        self.taskInfo = []
    
    def get_text(self):
        return "".join(["\n" for _ in self.taskInfo])

    def lex_document(self,document : Document) -> Callable[[int], StyleAndTextTuples]:
        def get_line(lineno : int) -> StyleAndTextTuples:
            try:
                task = self.taskInfo[lineno]
                style_base = "class:pending_task"
                if task["status"] == "Running":
                    style_base = "class:running_task"
                ID = task["ID"]
                GPU = str(task["GPU_ID"]) if task["GPU_ID"] != [-1] else " "
                description = task["description"]
                timestamp = task["timestamp"] if task["timestamp"] != "" else "                     "
                return [
                    (style_base + "_status" , "Status : " + task["status"] + " "),
                    (style_base +"_id" , " ID : " + ID + " "),
                    (style_base +"_gpu" , " GPU : " + GPU + " "),
                    (style_base +"_time" , " Folder : " + timestamp + " "),
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

        self.shower = MyTextArea(
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