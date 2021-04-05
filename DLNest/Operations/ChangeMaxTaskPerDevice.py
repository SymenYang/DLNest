from DLNest.Scheduler.Scheduler import Scheduler

def changeMaxTaskPerDevice(
    newMax : int
):
    scheduler = Scheduler()
    scheduler.changeMaxTaskPerDevice(newMax)