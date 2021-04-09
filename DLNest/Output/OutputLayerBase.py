from abc import ABCMeta, abstractmethod
import prompt_toolkit
from prompt_toolkit.styles import Style
import threading

class OutputLayerBase():
    def __init__(self):
        self.styleDict = {
            '' : '#efefef',
            'app' : '#ff0f9f',
            'status' : '#ff0000',
            'message' : '#ffffff', 
        }
        if not hasattr(self,'styledText'):
            self.styledText = []
        self.maxBufferLen = 10000
        self.offset = 0
        self.lock = threading.Lock()
        self.isLineHead = True

    def getPlainText(self,from_token : int = -1,length : int = -1):
        if self.lock.acquire():
            try:
                if len(self.styledText) == 0:
                    return self.offset,""

                if from_token == -1:
                    from_token = self.offset

                if self.offset > from_token:
                    # 开始位置已被删除
                    return self.offset,None
                else:
                    ret = ""
                    start = from_token - self.offset
                    end = 0
                    # 开始位置还没数据
                    if start >= len(self.styledText):
                        return self.offset,None

                    if length == -1:
                        end = len(self.styledText)
                    else:
                        end = min(len(self.styledText),start + length)

                    for i in range(start,end):
                        ret = ret + self.styledText[i][1]
                    return self.offset,ret
            finally:
                self.lock.release()               

    def getStyledText(self,from_token : int = -1,length : int = -1):
        if self.lock.acquire():
            try:
                if len(self.styledText) == 0:
                    return self.offset,[("","")]
                if from_token == -1:
                    from_token = self.offset

                if self.offset > from_token:
                    # 开始位置已被删除
                    return self.offset,None
                else:
                    start = from_token - self.offset
                    if start >= len(self.styledText):
                        # 开始位置还没数据
                        return self.offset,None
                    if length == -1:
                        end = len(self.styledText)
                    else:
                        end = min(len(self.styledText),start + length)

                    return self.offset,self.styledText[start:end]
            finally:
                self.lock.release()

    def getStyledDict(self):
        return self.styleDict

    def maintainText(self):
        if len(self.styledText) <= self.maxBufferLen:
            return
        while len(self.styledText) > self.maxBufferLen:
            while self.styledText[0][1] != "\n":
                self.styledText.pop(0)
                self.offset += 1
            self.styledText.pop(0)
            self.offset += 1

    def putPlainText(self,text : str):
        if self.lock.acquire():
            try:
                self.styledText.append((
                    (self.styleDict[''] , text)
                ))
                self.maintainText()
                return True
            finally:
                self.lock.release()
    
    def putStyledText(self,style_name : str, text : str):
        if self.lock.acquire():
            try:
                if style_name in self.styleDict:
                    self.styledText.append((
                        self.styleDict[style_name], text
                    ))
                    self.maintainText()
                    return True
                else:
                    return False
            finally:
                self.lock.release()

    def clear(self):
        if self.lock.acquire():
            try:
                self.offset = 0
                self.styledText = []
            finally:
                self.lock.release()

    def write(self,message : str):
        if self.isLineHead:
            self.putStyledText('app',"[" + self.appName + "] ")
            self.isLineHead = False

        if "\n" in message:
            if message == '\n':
                self.putStyledText('message',message)
                self.isLineHead = True
            else:
                lines = message.split("\n")
                endBy_n = message[-1] == "\n"
                for i in range(len(lines)):
                    if self.isLineHead:
                        self.putStyledText('app',"[" + self.appName + "] ")
                        self.isLineHead = False
                    self.putStyledText('message',lines[i])
                    if i < len(lines) - 1:
                        self.putStyledText('message',"\n")
                        self.isLineHead = True
                    elif endBy_n:
                        self.putStyledText('message',"\n")
                        self.isLineHead = True
        else:
            self.putStyledText('message',message)

    def flush(self):
        ...

    def isatty(self):
        return True

    def fileno(self):
        return 3 # Maybe hazard