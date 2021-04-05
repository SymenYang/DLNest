import pytest
from DLNest.Information.AnalyzeTask import AnalyzeTask
import time
from pathlib import Path

@pytest.mark.Information
class TestAnalyzeTask:
    def test_init(self):
        AT = AnalyzeTask(
            recordPath = "/root/code/DLNestTest/Saves/Some Description",
            scriptPath = "scriptPath",
            checkpointID = 233
        )
        assert isinstance(AT.recordPath,Path)
        assert isinstance(AT.scriptPath,Path)
        assert str(AT.recordPath) == "/root/code/DLNestTest/Saves/Some Description"
        assert str(AT.scriptPath) == "scriptPath"
        assert AT.checkpointID == 233
        assert AT.type == "Analyze"
        assert AT.commandPipe == None
        assert AT.outputQueue == None
    
    def test_getDict(self):
        AT = AnalyzeTask(
            recordPath = "/root/code/DLNestTest/Saves/Some Description",
            scriptPath = "scriptPath",
            checkpointID = 233
        )
        retDict = AT.getDict()
        assert retDict["record_path"] == "/root/code/DLNestTest/Saves/Some Description"
        assert retDict["script_path"] == "scriptPath"
        assert retDict["checkpoint_ID"] == 233