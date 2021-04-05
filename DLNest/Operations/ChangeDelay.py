from DLNest.Scheduler.Scheduler import Scheduler

def changeDelay(
    newDelay : int
):
    scheduler = Scheduler()
    scheduler.changeTimeDelay(newDelay)