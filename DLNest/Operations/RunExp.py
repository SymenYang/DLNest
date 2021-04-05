from DLNest.Information.InfoCenter import InfoCenter
from DLNest.Information.AnalyzeTask import AnalyzeTask
from DLNest.Output.AnalyzerBuffer import AnalyzerBuffer

def runExp(
    taskID : str,
    command : str
):
    infoCenter = InfoCenter()
    analyzeTask : AnalyzeTask = infoCenter.getTaskByID(taskID)
    
    assert analyzeTask != None # Wrong task
    assert isinstance(analyzeTask,AnalyzeTask) # Not an analyze task
    assert isinstance(analyzeTask.outputBuffer,AnalyzerBuffer) # don't have analyze buffer
    assert analyzeTask.process != None # not running
    assert analyzeTask.process.is_alive() # not alive
    assert analyzeTask.commandQueue != None # don't have command queue

    analyzeTask.commandQueue.put(command,block = False)