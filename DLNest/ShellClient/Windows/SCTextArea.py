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

class SCTextArea(TextArea):
    def __init__(self,lexer,wrap_lines = True):
        super(SCTextArea,self).__init__(
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