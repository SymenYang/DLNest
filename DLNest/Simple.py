import requests
import argparse
from prompt_toolkit import PromptSession,HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import traceback
import json
from DLNest.ShellClient.Communicator import Communicator


class DLNestSimple:
    def __init__(self, url : str = "127.0.0.1", port : int = 9999):
        self.communicator = Communicator(url = url, port = port)
    def run(self):
        self.session = PromptSession(auto_suggest = AutoSuggestFromHistory())
        while True:
            try:
                command = self.session.prompt(HTML("<seagreen><b>DLNest>></b></seagreen>"))
                commandWordList = command.strip().split(' ')
                output = self.communicator.giveACommand(commandWordList)
                print(output)
                if "exit" in output:
                    exit(0)
            except KeyboardInterrupt:
                exit(0)
            except Exception as e:
                s = traceback.format_exc()
                listS = s.split("\n")[:-1]
                s = "\n".join(listS[-3:])
                print(s)

if __name__ == "__main__":
    main = DLNestSimple()
    main.run()