from DLNest.Information.InfoCenter import InfoCenter
from DLNest.Information.AnalyzeTask import AnalyzeTask
from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer

def getAnalyzeOutput(
    taskID : str,
    style : bool = True
):
    infoCenter = InfoCenter()
    analyzeTask : AnalyzeTask = infoCenter.getTaskByID(taskID)
    
    assert analyzeTask != None # Wrong task
    assert isinstance(analyzeTask,AnalyzeTask) # Not an analyze task
    assert isinstance(analyzeTask.outputBuffer,AnalyzerBuffer) # don't have analyze buffer
    assert analyzeTask.process != None # not running
    assert analyzeTask.process.is_alive() # not alive

    if style:
        return analyzeTask.outputBuffer.getStyledText()
    else:
        return analyzeTask.outputBuffer.getPlainText()