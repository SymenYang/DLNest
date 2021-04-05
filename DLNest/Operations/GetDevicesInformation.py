from DLNest.Information.InfoCenter import InfoCenter

def getDevicesInformation():
    infoCenter = InfoCenter()
    return infoCenter.getAvailableDevicesInformation()