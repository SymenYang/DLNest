from DLNest.Operations.AnalyzeIndependent import AnalyzeIndependent

import os
import pynvml
import argparse
from pathlib import Path
import importlib

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class AnalyzeArguments(Arguments):
    def __init__(self):
        super(AnalyzeArguments, self).__init__(desc = "Arguments for DLNest analyze.")

        self._parser.add_argument("-r",type=str, help = "record path", required=True)
        self._parser.add_argument("-s",type=str, help = "script path", required=True)
        self._parser.add_argument("-c",type=int, default=-1,help = "checkpoint ID")
        self._parser.add_argument("-devices",default = [],nargs='+', type=int)
        self._parser.add_argument("-CPU",action='store_true',help="Set to use CPU.")

def _loadAScript(filePath : Path,name : str):
    # load a script by name
    spec = importlib.util.spec_from_file_location(
        name,
        filePath
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _analyze(args):
    if args.CPU:
        devices = [-1]
    else:
        devices = args.devices
    if devices == []:
        devices = [0]

    scriptModule = _loadAScript(Path(args.s),"AScript")
    expFunc = scriptModule.__getattribute__("experience")

    AnalyzeIndependent(
        recordPath = args.r,
        expFunc = expFunc,
        scriptPath = "",
        checkpointID = args.c,
        devices = devices
    )

def analyze(
    recordPath : str,
    checkpointID : int = -1,
    scriptPath : str = "",
    devices : [int] = [0],
    expFunc = None
):
    AnalyzeIndependent(
        recordPath = recordPath,
        expFunc = expFunc,
        scriptPath = scriptPath,
        devices = devices,
        checkpointID = checkpointID
    )

if __name__ == "__main__":
    argparser = AnalyzeArguments()
    parser = argparser.parser()
    args = parser.parse_args()
    _analyze(args)