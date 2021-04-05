from prompt_toolkit.widgets import Frame,TextArea,Label
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from typing import Callable, Iterable, List, Optional
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import VSplit, Window,HSplit
try:
    from .utils.Completers import getCommandCompleter
except ImportError:
    from utils.Completers import getCommandCompleter
import os

class CommandInput:
    def __init__(self,title: str = "DLNest Command Line",onAccept : Callable[[str],None] = None):
        self.kb = KeyBindings()
        self.onAccept = onAccept

        def accept_text(buf : Buffer):
            if not self.onAccept is None:
                self.onAccept(buf.text)
            return False

        self.completer = getCommandCompleter()
        self.title = title

        self.text = TextArea(
            height=3,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            prompt=[("#fff5ee","DLNest>>")],
            accept_handler=accept_text,
            scrollbar=True,
            multiline=False
        )
        self.text.height=3
        self.infoBar = Label(
            [("bg:#006060","Press Enter to enter a command. Press ctrl + c to exit.")]
        )

        self.content = HSplit([
            self.text,
            self.infoBar
        ])

    def getWindow(self):
        return Frame(
            self.content,
            title = self.title,
            style="class:command_frame"
        )