from prompt_toolkit.completion import CompleteEvent, Completer, Completion,merge_completers,WordCompleter
from prompt_toolkit.completion.nested import NestedCompleter
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from typing import Callable, Iterable, List, Optional
from prompt_toolkit.document import Document
import os

class PartPathCompleter(Completer):
    """
    Complete for Path variables.

    :param get_paths: Callable which returns a list of directories to look into
                      when the user enters a relative path.
    :param file_filter: Callable which takes a filename and returns whether
                        this file should show up in the completion. ``None``
                        when no filtering has to be done.
    :param min_input_len: Don't do autocompletion when the input string is shorter.
    """

    def __init__(
        self,
        only_directories: bool = False,
        get_paths: Optional[Callable[[], List[str]]] = None,
        file_filter: Optional[Callable[[str], bool]] = None,
        min_input_len: int = 0,
        expanduser: bool = False,
    ) -> None:

        self.only_directories = only_directories
        self.get_paths = get_paths or (lambda: ["."])
        self.file_filter = file_filter or (lambda _: True)
        self.min_input_len = min_input_len
        self.expanduser = expanduser

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        text = text.split(" ")[-1]

        # Complete only when we have at least the minimal input length,
        # otherwise, we can too many results and autocompletion will become too
        # heavy.
        if len(text) < self.min_input_len:
            return

        try:
            # Do tilde expansion.
            if self.expanduser:
                text = os.path.expanduser(text)

            # Directories where to look.
            dirname = os.path.dirname(text)
            if dirname:
                directories = [
                    os.path.dirname(os.path.join(p, text)) for p in self.get_paths()
                ]
            else:
                directories = self.get_paths()

            # Start of current file.
            prefix = os.path.basename(text)

            # Get all filenames.
            filenames = []
            for directory in directories:
                # Look for matches in this directory.
                if os.path.isdir(directory):
                    for filename in os.listdir(directory):
                        if filename.startswith(prefix):
                            filenames.append((directory, filename))

            # Sort
            filenames = sorted(filenames, key=lambda k: k[1])

            # Yield them.
            for directory, filename in filenames:
                completion = filename[len(prefix) :]
                full_name = os.path.join(directory, filename)

                if os.path.isdir(full_name):
                    # For directories, add a slash to the filename.
                    # (We don't add them to the `completion`. Users can type it
                    # to trigger the autocompletion themselves.)
                    filename += "/"
                elif self.only_directories:
                    continue

                if not self.file_filter(full_name):
                    continue

                yield Completion(completion, 0, display=filename)
        except OSError:
            pass

class CommandCompleter(Completer):
    def __init__(self, rules : dict):
        self.rules = rules

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ):
        text = document.text_before_cursor.lstrip()
        tokens = text.replace("\n"," ").replace("\t"," ").split(" ")
        nowRules = self.rules
        if len(tokens) <= 1:
            completer = WordCompleter(
                list(self.rules.keys())
            )
            for c in completer.get_completions(document, complete_event):
                yield c
        else:
            title = tokens[0]
            if not title in self.rules:
                return
            else:
                try:
                    rule_dict = self.rules[title]
                    if len(tokens) > 2 and tokens[-2] in rule_dict:
                        if not isinstance(rule_dict[tokens[-2]],Completer):
                            if not isinstance(rule_dict[tokens[-2]],int):
                                return
                        else:
                            for c in rule_dict[tokens[-2]].get_completions(document, complete_event):
                                yield c
                            return
                    search_list = []
                    if rule_dict is None:
                        return
                    for key in rule_dict:
                        if not key in tokens:
                            search_list.append(key)
                    completer = WordCompleter(
                        search_list
                    )
                    for c in completer.get_completions(document, complete_event):
                        yield c
                except Exception as e:
                    ...
                    #print(e)

def getCommandCompleter():
    commands = {
        "run" : {
            "-c" : PartPathCompleter(),
            "-d" : None,
            "-f" : PartPathCompleter(),
            "-m" : None,
            "-ns" : -1,
            "-mc" : -1,
            "-sd" : -1,
            "-DDP" : -1,
            "-CPU" : -1
        },
        "continue" : {
            "-r" : PartPathCompleter(),
            "-c" : None,
            "-d" : None,
            "-m" : None,
            "-DDP" : -1,
            "-CPU" : -1,
            "-mc" : -1
        },
        "analyze" : {
            "-r" : PartPathCompleter(),
            "-s" : PartPathCompleter(),
            "-c" : None,
            "-m" : None,
            "-CPU" : -1
        },
        "new" : {
            "-d" : PartPathCompleter(),
        },
        "changeDevices" : {
            "-d" : None
        },
        "changeDelay" : None,
        "runExp" : None,
        "del" : None,
        "exit" : None
    }
    return CommandCompleter(commands)