from DLNest.Output.OutputLayerBase import OutputLayerBase

from multiprocessing import Queue

class AnalyzerBuffer(OutputLayerBase):
    def __init__(self):
        super(AnalyzerBuffer,self).__init__()
        self.styleDict = {
            '' : '#efefef',
            'app' : '#2afa38 bold',
            'ignored':'#ff7070',
            'error' : '#ff0000',
            'message' : '#efefef', 
        }
        self.appName = "DLNest Analyzer"
        self.outputQueue = Queue()
        self.isSend = False

    def logMessage(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
        self.sendData()

    def logIgnError(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('ignored' , "[Ignored] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
        self.sendData()

    def logError(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('error' , "[ERROR] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
        self.sendData()

    def logDebug(self,message : str):
        self.putStyledText('app',"[" + self.appName + "] ")
        self.putStyledText('ignored' , "[DEBUG] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
        self.sendData()

    def write(self,message : str):
        super().write(message)
        self.sendData()

    def sendData(self):
        # Temproral implementation. Trans all styled text to others.
        if not self.isSend:
            return
        if self.lock.acquire():
            try:
                while not self.outputQueue.empty():
                    try:
                        self.outputQueue.get(block=False)
                    except Exception as e:
                        pass

                self.outputQueue.put(self.styledText,block=False)
            finally:
                self.lock.release()

    def receiveData(self):
        if self.isSend:
            return
        if self.lock.acquire():
            try:
                styledText = self.outputQueue.get(block = False)
                self.styledText = styledText
            except Exception as e:
                pass
            finally:
                self.lock.release()
    
    def getPlainText(self,from_token : int = -1,length : int = -1):
        self.receiveData()
        return super().getPlainText(from_token = from_token, length = length)
    
    def getStyledText(self,from_token : int = -1,length : int = -1):
        self.receiveData()
        return super().getStyledText(from_token = from_token, length = length)