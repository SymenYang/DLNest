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

def testTask(self):
    self.count += 1
    self.lexer.styled_text += [("#ff0000","test" + str(self.count)),("","\n")]
    try:
        self.shower.text = self.shower.text + "test" + str(self.count) + "\n"
    except Exception as e:
        pass

class styledTextLexer(Lexer):
    def __init__(self,styled_text : StyleAndTextTuples = []):
        self.styled_text = styled_text

    def __get_styled_lines(self):
        self.styled_text_lines = []
        now = []
        for item in self.styled_text:
            finish = False
            if "\n" in item[1]:
                finish = True
                item = (item[0],item[1].replace("\n", ""))
            now.append(item)
            if finish:
                self.styled_text_lines.append(now)
                now = []
        self.styled_text_lines.append(now)

    def lex_document(self,document : Document) -> Callable[[int], StyleAndTextTuples]:
        self.__get_styled_lines()
        lines = document.lines
        def get_line(lineno : int) ->StyleAndTextTuples:
            try:
                return self.styled_text_lines[lineno]
            except Exception:
                return [("","")]
        
        return get_line

class MyTextArea(TextArea):
    def __init__(self,lexer,wrap_lines = True):
        super(MyTextArea,self).__init__(
            lexer = lexer,
            read_only=True,
            focusable=True,
            scrollbar=True,
            wrap_lines=wrap_lines
        )

    @property
    def text(self) -> str:
        """
        The `Buffer` text.
        """
        return self.buffer.text

    @text.setter
    def text(self, value: str) -> None:
        oldPos = self.document.cursor_position
        if len(self.document.text) == oldPos:
            self.document = Document(value)#, 0)
        else:
            self.document = Document(value,oldPos)

class ResultsOutput:
    def __init__(
        self,
        title : str = "DLNest Output",
        routineTask = None,
        freq : int = 1,
        style : str = "class:results_output"
    ):
        self.title = title
        self.routineTask = routineTask
        self.scheduler = BackgroundScheduler()
        if not routineTask is None:
            self.scheduler.add_job(self.routineTask,'interval',seconds=freq,args=[self])
        self.scheduler.start()
        self.lexer = styledTextLexer()

        self.shower = MyTextArea(
            self.lexer
        )

        self.style = style
        self.setKeyBinding()
        self.window = Frame(
            self.shower,
            self.title,
            style=self.style,
            modal=True,
            key_bindings=self.kb
        )

    def setKeyBinding(self):
        self.kb = KeyBindings()

        @self.kb.add("escape")
        def toEnd(event):
            self.shower.buffer.cursor_position = len(self.shower.document.text)
        


    def getWindow(self):
        return self.window

class AnalyzerOutput(ResultsOutput):
    def __init__(
        self,
        title : str = "DLNest Output",
        routineTask = None,
        analyzerRoutineTask = None,
        freq : int = 1,
        analyzerFreq : int = 5,
        style : str = "class:analyzer_output"
    ):
        super(AnalyzerOutput,self).__init__(
            title,
            routineTask,
            freq,
            style
        )
        self.analyzerRoutineTask = analyzerRoutineTask
        if not self.analyzerRoutineTask is None:
            self.scheduler.add_job(self.analyzerRoutineTask,'interval',seconds=analyzerFreq,args=[self])

        self.infoText = FormattedTextControl(
                    [("","No analyzer task is running")],
                    focusable=False,
                    show_cursor=False
                )
        
        self.infoWindow = Window(
                content=self.infoText,
                width=27
            )

        self.infoLabel = Box(
            body=self.infoWindow,
            height=3,
            padding_top=1,
            padding_bottom=1,
            padding_left=3,
            padding_right=3,
            style="class:analyzer_info_label"
        )

        self.window = Frame(
            HSplit([
                self.infoLabel,
                self.shower
            ]),
            title=self.title,
            style = self.style,
            modal = True,
            key_bindings=self.kb
        )