class TrainStdout:
    def __init__(self,fp,showOnScreen = False,originalStdout = None):
        super(TrainStdout,self).__init__()
        self.stdout = fp
        self.screenout = originalStdout
        self.showOnScreen = showOnScreen
    
    def write(self,s):
        ret = self.stdout.write(s)
        self.stdout.flush()
        if self.showOnScreen:
            self.screenout.write(s)
        return ret

    def flush(self):
        if self.showOnScreen:
            self.screenout.flush()
        return self.stdout.flush()

    def isatty(self):
        return self.stdout.isatty()
    
    def fileno(self):
        return self.stdout.fileno()