import smtplib
from email.mime.text import MIMEText
from email.header import Header
import logging

_fromTitle = "DLNest Mail Plugin"
_toTitle = "DLNest User"
def sendSelfMail(username : str, password : str, message : str, subject : str, host : str, port : int = 25, fromTitle : str = _fromTitle, toTitle : str = _toTitle):
    message = MIMEText(message,'plain', 'utf-8')
    message["From"] = Header(fromTitle, 'utf-8')
    message["To"] = Header(toTitle, 'utf-8')
    message["subject"] = Header(subject,'utf-8')
    try:
        smtpObj = smtplib.SMTP(host = host, port = port)
        smtpObj.login(username, password)
        smtpObj.sendmail(username, [username], message.as_string())
        logging.info("[SendSelfMail] Send mail successfully to {}".format(username))
    except Exception as e:
        logging.debug("[SendSelfMail]" + str(e))

def sendMail(username : str, password : str, toName : str, message : str, subject : str, host : str, port : int = 25, fromTitle : str = _fromTitle, toTitle : str = _toTitle):
    message = MIMEText(message,'plain', 'utf-8')
    message["From"] = Header(fromTitle, 'utf-8')
    message["To"] = Header(toTitle, 'utf-8')
    message["subject"] = Header(subject,'utf-8')

    receiver = toName if isinstance(toName, list) else [toName]

    try:
        smtpObj = smtplib.SMTP(host = host, port = port)
        smtpObj.login(username, password)
        smtpObj.sendmail(username, receiver, message.as_string())
        logging.info("[SendMail] Send mail successfully to {}".format(username))
    except Exception as e:
        logging.debug("[SendMail]" + str(e))