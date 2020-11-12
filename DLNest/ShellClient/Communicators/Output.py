import requests
import argparse
import json

class OutputCommunicator:
    def __init__(self,url : str):
        self.url = url
    
    def getOutput(self,type = "styled"):
        try:
            return requests.get(self.url,{
                "type" : type
            })
        except Exception:
            return None