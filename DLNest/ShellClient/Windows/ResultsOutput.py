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

class styledTextLexer(Lexer):
    def __init__(self,styled_text : StyleAndTextTuples = []):
        self.styled_text = styled_text
        self.DEBUG = False

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

        if self.DEBUG:
            with open("/root/STLexerDEBUG.txt","w") as f:
                print(self.styled_text,file = f)
                print(self.styled_text_lines,file=f)
                print(lines,file = f)
                print(len(lines),len(self.styled_text_lines),file=f)

        def get_line(lineno : int) -> StyleAndTextTuples:
            try:
                return self.styled_text_lines[lineno]
            except Exception:
                return [("","")]
        
        return get_line

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

        self.shower = SCTextArea(
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

class AnalyzeOutput(ResultsOutput):
    def __init__(
        self,
        title : str = "DLNest Output",
        routineTask = None,
        freq : int = 1,
        style : str = "class_analyzer_output"
    ):
        super(AnalyzeOutput,self).__init__(
            title,
            routineTask,
            freq,
            style
        )
        
        self.infoText = FormattedTextControl( 
                        [("",   "           No analyze task is running           ")],
                        focusable=False,
                        show_cursor=False
        )

        self.infoWindow = Window(
                    content=self.infoText
        )

        self.infoLabel = Box(
            body=self.infoWindow,
            height=3,
            padding_top=1,
            padding_bottom=1,
            # padding_left=3,
            # padding_right=3,
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