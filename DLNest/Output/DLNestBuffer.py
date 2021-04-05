from DLNest.Output.OutputLayerBase import OutputLayerBase

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance

class DLNestBuffer(OutputLayerBase,Singleton):
    def __init__(self):
        if hasattr(self,"styledText"):
            return
        OutputLayerBase.__init__(self)
        self.styleDict = {
            '' : '#efefef',
            'app' : '#ff0f9f bold',
            'ignored':'#ff7070',
            'error' : '#ff0000',
            'message' : '#efefef', 
        }
        self.appName = "DLNest"
    
    def logMessage(self,message : str):
        self.putStyledText('app',"["+ self.appName +"] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logIgnError(self,message : str):
        self.putStyledText('app',"["+ self.appName +"] ")
        self.putStyledText('ignored' , "[Ignored] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
    
    def logError(self,message : str):
        self.putStyledText('app',"["+ self.appName +"] ")
        self.putStyledText('error' , "[ERROR] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')

    def logDebug(self,message : str):
        self.putStyledText('app',"["+ self.appName +"] ")
        self.putStyledText('ignored' , "[DEBUG] ")
        self.putStyledText('message',message)
        self.putStyledText('message','\n')
