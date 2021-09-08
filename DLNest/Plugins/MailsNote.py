from DLNest.Plugins.DLNestPluginBase import DLNestPluginBase as DPB
from DLNest.Plugins.Utils.SendMailTools import sendSelfMail
import DLNest
import traceback
import logging

class DLNestPlugin(DPB):
    _NAME = "MailsNote"
    _config = {
        "enable_list" : ["Aborting","TrainFinish"],
        "username" : "",
        "password" : "",
        "host" : "mail.fudan.edu.cn",
        "port" : 25
    }
    _defaultKeys = ["enable_list","username","password","host"]
    
    def trainAborting(self, exception : Exception):
        excStr = traceback.format_exc()

        pluginName = DLNestPlugin._NAME
        enable = "Aborting" in DPB.getArgs(self, pluginName, "enable_list", [])
        if not enable:
            logging.debug("MailsNote is not enabled for aborting")
            return 
        
        username = DPB.getArgs(self, pluginName, "username", "")
        password = DPB.getArgs(self, pluginName, "password", "")
        host = DPB.getArgs(self, pluginName, "host", "mail.fudan.edu.cn")
        port = DPB.getArgs(self, pluginName, "port", 25)
        message = "Train task in {} aborted with this exception\n".format(self.getArgs()["root_file_path"]) + excStr
        subject = "Train Aborted!"
        sendSelfMail(
            username = username,
            password = password,
            message = message,
            subject = subject,
            host = host,
            port = port
        )
        logging.info("[MailsNote] " + message)
    
    def ATrain(self):
        pluginName = DLNestPlugin._NAME
        enable = "TrainFinish" in DPB.getArgs(self, pluginName, "enable_list", [])
        if not enable:
            logging.debug("MailsNote is not enabled for train finish")
            return

        username = DPB.getArgs(self, pluginName, "username", "")
        password = DPB.getArgs(self, pluginName, "password", "")
        host = DPB.getArgs(self, pluginName, "host", "mail.fudan.edu.cn")
        port = DPB.getArgs(self, pluginName, "port", 25)
        message = "Train task in {} finished\n".format(self.getArgs()["root_file_path"])

        resultStr = DLNest.Plugins.MailsNote.DLNestPlugin._getResultsStr() # To get correct class function and var.
        if len(resultStr) != 0:
            message += "\nThe final results are: \n" + resultStr

        subject = "Train Finished!"
        sendSelfMail(
            username = username,
            password = password,
            message = message,
            subject = subject,
            host = host,
            port = port
        )
        logging.info("[MailsNote] " + message)

    _result_values = {}

    @classmethod
    def giveResultValues(cls, resultsDict : dict):
        cls._result_values = resultsDict
    
    @classmethod
    def _getResultsStr(cls):
        str = ""
        for key in cls._result_values:
            str += "| {} : {}".format(key, cls._result_values[key])
        return str
    
    def custom(self, message, subject):
        pluginName = DLNestPlugin._NAME
        username = DPB.getArgs(self, pluginName, "username", "")
        password = DPB.getArgs(self, pluginName, "password", "")
        host = DPB.getArgs(self, pluginName, "host", "mail.fudan.edu.cn")
        port = DPB.getArgs(self, pluginName, "port", 25)
        message = "Train task in {} gives a message\n".format(self.getArgs()["root_file_path"]) + message
        
        sendSelfMail(
            username = username,
            password = password,
            message = message,
            subject = subject,
            host = host,
            port = port
        )
        logging.info("[MailsNote] " + message)
    
    def SOTA(self, key, value):
        value = str(value)
        message = "Congratulations! SOTA performance in {} has been made by your train task {} with value {}!".format(key, self.getArgs()["root_file_path"], value)
        subject = "SOTA for {}!".format(key)
        DLNestPlugin.custom(self, message, subject)