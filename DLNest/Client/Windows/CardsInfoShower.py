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

class cardsLexer(Lexer):
    def __init__(self):
        self.cardsInfo = []
        self.freeMemoryLength = 8
        self.runningTasksLength = 3
    
    def get_text(self):
        return "".join(["\n" for _ in self.cardsInfo])

    def lex_document(self,document : Document) -> Callable[[int], StyleAndTextTuples]:
        def get_line(lineno : int) -> StyleAndTextTuples:
            try:
                card = self.cardsInfo[lineno]
                ID = card["real_ID"]
                isBreak = "Break" if card["is_break"] else "Valid"
                break_class = "break" if card["is_break"] else "valid"
                freeMemory = str(int(card["free_memory"])) + " MB"
                runningTasks = str(len(card["running_tasks"]))

                if len(freeMemory) < self.freeMemoryLength:
                    freeMemory = " " * (self.freeMemoryLength - len(freeMemory)) + freeMemory
                
                if len(runningTasks) < self.runningTasksLength:
                    runningTasks = " " * (self.runningTasksLength - len(runningTasks)) + runningTasks

                return [
                    ("class:cards_id"                   ,"Cards: " + str(ID) + " "),
                    ("class:cards_status_" + break_class," " + isBreak + " "),
                    ("class:cards_free_memory"          ," F-Memory: " + freeMemory + " "),
                    ("class:cards_tasks"                ," #Tasks: " + runningTasks + " ")
                ]
            except Exception as e:
                return []
        
        return get_line

class CardsInfoShower:
    def __init__(
        self, 
        title : str = "Tasks",
        routineTask = None,
        freq : int = 1,
        style : str = "class:cards_info_shower"
    ):

        self.title = title
        self.routineTask = routineTask
        self.scheduler = BackgroundScheduler()
        if not routineTask is None:
            self.scheduler.add_job(self.routineTask,'interval',seconds=freq,args=[self])
        self.scheduler.start()
        self.lexer =cardsLexer()

        self.shower = MyTextArea(
            lexer = self.lexer,
            wrap_lines=False
        )

        self.style = style
        self.window = Frame(
            self.shower,
            self.title,
            style = self.style,
            height=10,
            width=55
        )
    
    def getWindow(self):
        return self.window