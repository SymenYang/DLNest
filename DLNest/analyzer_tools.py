from DLNest.Core.DLNestCore import DLNestCore
from DLNest.Core.AnalyzeProcess import AnalyzeProcess,AnalyzeRunner
from DLNest.BridgeLayers.InformationLayer import AnalyzeTask
import json
from pathlib import Path
import argparse
import time

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class AnalyzeArguments(Arguments):
    def __init__(self):
        super(AnalyzeArguments, self).__init__(desc="Arguments for an Analyzer")

        self._parser.add_argument("-r",type=str, help = "path to the model record directory.")
        self._parser.add_argument("-s",type=str, help = "path to the analyze scripts.")
        self._parser.add_argument("-c",type=int, help = "which epoch you want the model to load.(int)")
        self._parser.add_argument("-m",type=int, default = -1, help="predicted GPU memory consumption for this task in MB.(default: 90\% of the total memory)")

def load_env():
    argparser = AnalyzeArguments()
    parser = argparser.parser()
    args = parser.parse_args()
    if args.s is None or args.s == "":
        args.s = Path(args.r).parent.parent / "AnalyzeScripts"
        args.s = str(args.s)
    task = AnalyzeTask(
        args.r,
        args.s,
        args.c,
        0,
        args.m
    )
    analyzeProcess = AnalyzeProcess(task,None,None)
    analyzeProcess._AnalyzeProcess__initAll()
    return analyzeProcess.runner

if __name__ == "__main__":
    runner = load_env()
    print(runner.model,runner.args,runner.dataset)
