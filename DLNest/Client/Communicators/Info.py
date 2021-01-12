import requests
import argparse
import json

class InfoCommunicator:
    def __init__(self,url : str):
        self.url = url
    
    def getTaskInfo(self):
        try:
            return requests.get(self.url + "/task_info")
        except Exception:
            return None

    def getAnalyzerTaskInfo(self):
        try:
            return requests.get(self.url + "/analyze_task_info")
        except Exception:
            return None

    def getCardsInfo(self):
        try:
            return requests.get(self.url + "/cards_info")
        except Exception:
            return None