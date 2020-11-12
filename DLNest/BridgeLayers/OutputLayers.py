import prompt_toolkit
from prompt_toolkit.styles import Style
import threading
import io
import sys

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

class OutputLayer(Singleton):
    def __init__(self):
        self.style_dict = {
            '' : '#efefef',
            'app' : '#ff0f9f',
            'status' : '#ff0000',
            'message' : '#ffffff', 
        }
        if not hasattr(self,'styled_text'):
            self.styled_text = []
        self.max_buffer_len = 10000
        self.offset = 0
        self.lock = threading.Lock()
    
    def getPlainText(self,from_token : int = -1,length : int = -1):
        if self.lock.acquire():
            try:
                if len(self.styled_text) == 0:
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
                    if start >= len(self.styled_text):
                        return self.offset,None

                    if length == -1:
                        end = len(self.styled_text)
                    else:
                        end = min(len(self.styled_text),start + length)

                    for i in range(start,end):
                        ret = ret + self.styled_text[i][1]
                    return self.offset,ret
            finally:
                self.lock.release()               

    def getStyledText(self,from_token : int = -1,length : int = -1):
        if self.lock.acquire():
            try:
                if len(self.styled_text) == 0:
                    return self.offset,[("","")]
                if from_token == -1:
                    from_token = self.offset

                if self.offset > from_token:
                    # 开始位置已被删除
                    return self.offset,None
                else:
                    start = from_token - self.offset
                    if start >= len(self.styled_text):
                        # 开始位置还没数据
                        return self.offset,None
                    if length == -1:
                        end = len(self.styled_text)
                    else:
                        end = min(len(self.styled_text),start + length)

                    return self.offset,self.styled_text[start:end]
            finally:
                self.lock.release()

    def getStyledDict(self):
        return self.style_dict

    def maintainText(self):
        if len(self.styled_text) <= self.max_buffer_len:
            return
        while len(self.styled_text) > self.max_buffer_len:
            while self.styled_text[0][1] != "\n":
                self.styled_text.pop(0)
                self.offset += 1
            self.styled_text.pop(0)
            self.offset += 1

    def putPlainText(self,text : str):
        if self.lock.acquire():
            try:
                self.styled_text.append((
                    (self.style_dict[''] , text)
                ))
                self.maintainText()
                return True
            finally:
                self.lock.release()
    
    def putStyledText(self,style_name : str, text : str):
        if self.lock.acquire():
            try:
                if style_name in self.style_dict:
                    self.styled_text.append((
                        self.style_dict[style_name], text
                    ))
                    self.maintainText()
                    return True
                else:
                    return False
            finally:
                self.lock.release()

class DLNestBuffer(OutputLayer):
    def __init__(self):
        super(DLNestBuffer,self).__init__()
        self.style_dict = {
            '' : '#efefef',
            'app' : '#ff0f9f bold',
            'ignored':'#ff7070',
            'error' : '#ff0000',
            'message' : '#efefef', 
        }
        #self.max_buffer_len = 50
    
    def logMessage(self,message : str,app : str = "DLNest"):
        self.putStyledText('app',"["+ app +"] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logIgnError(self,message : str,app : str = "DLNest"):
        self.putStyledText('app',"["+ app +"] ")
        self.putStyledText('ignored' , "[Ignored] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
    
    def logError(self,message : str,app : str = "DLNest"):
        self.putStyledText('app',"["+ app +"] ")
        self.putStyledText('error' , "[ERROR] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logDebug(self,message : str,app : str = "DLNest"):
        self.putStyledText('app',"["+ app +"] ")
        self.putStyledText('ignored' , "[DEBUG] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

class AnalyzerBuffer(OutputLayer,io.TextIOWrapper):
    def __init__(self):
        super(AnalyzerBuffer,self).__init__()
        self.style_dict = {
            '' : '#efefef',
            'app' : '#2afa38 bold',
            'ignored':'#ff7070',
            'error' : '#ff0000',
            'message' : '#efefef', 
        }
        self.appName = "DLNest Analyzer"

    def write(self,message : str):
        self.putStyledText('message',message)

    def flush(self):
        ...

    def isatty(self):
        return True
    
    def fileno(self):
        return 3

    def logMessage(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logIgnError(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('ignored' , "[Ignored] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
    
    def logError(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('error' , "[ERROR] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logDebug(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('ignored' , "[DEBUG] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

class TrainStdout:
    def __init__(self,fp):
        super(TrainStdout,self).__init__()
        self.stdout = fp
    
    def write(self,s):
        ret = self.stdout.write(s)
        self.stdout.flush()
        return ret

    def flush(self):
        return self.stdout.flush()

    def isatty(self):
        return self.stdout.isatty()
    
    def fileno(self):
        return self.stdout.fileno()