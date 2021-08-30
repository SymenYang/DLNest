from DLNest.Operations.RunIndependent import runIndependent

import os
import pynvml
import argparse

class Arguments:
    def __init__(self,desc : str = ""):
        self._parser = argparse.ArgumentParser(description=desc)

    def parser(self):
        return self._parser

class TaskArguments(Arguments):
    def __init__(self):
        super(TaskArguments, self).__init__(desc="Arguments for DLNest task.")

        self._parser.add_argument("-c",type=str, help="root configuration json file for this task.",required=True)
        self._parser.add_argument("-d",type=str, default = "", help="description for this task.(default: None)")
        self._parser.add_argument("-f",type=str, default = "", help="frequently changing configuration json file for this task.(default:None)")
        self._parser.add_argument("-devices",default = [],nargs='+', type=int)
        self._parser.add_argument("-ns",action='store_true', help="Set to save to the NOSAVE dir.")
        self._parser.add_argument("-ss",action='store_false', help="Set to also show on screen.")
        self._parser.add_argument("-DDP",action='store_true',help="Set to use DDP.")
        self._parser.add_argument("-CPU",action='store_true',help="Set to use CPU.")
        self._parser.add_argument("-sd",action='store_true',help="Set to use description as the save dir name.(coverd by ns)")

def runTrain(args):
    assert not(args.DDP and args.CPU)
    if args.CPU:
        devices = [-1]
    else:
        devices = args.devices
    if devices == []:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        devices = [i for i in range(count)]
    runIndependent(
        configPath = args.c,
        freqPath = args.f,
        devices = devices,
        description = args.d,
        DDP = args.DDP,
        noSave = args.ns,
        useDescriptionToSave = args.sd,
        showOnScreen = args.ss
    )

if __name__ == "__main__":
    import sys
    if sys.path[0] != '':
        sys.path[0] = ''
    argparser = TaskArguments()
    parser = argparser.parser()
    args = parser.parse_args()
    runTrain(args)
