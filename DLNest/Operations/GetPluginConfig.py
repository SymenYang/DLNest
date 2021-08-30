from pathlib import Path
import sys
import importlib
from DLNest.Plugins.DLNestPluginBase import DLNestPluginBase

def _loadAModule(filePath : Path,name : str):
    spec = importlib.util.spec_from_file_location(
        name,
        filePath
    )
    module = importlib.util.module_from_spec(spec)
    dirName = str(filePath.parent)
    if not dirName in sys.path:
        sys.path.append(dirName)
    spec.loader.exec_module(module)
    return module

def _loadAPlugin(pluginName : str):
    pluginPath = Path(__file__).parent.parent / "Plugins" / (pluginName + '.py')
    tmpPath = Path(pluginName)
    if tmpPath.is_absolute():
        pluginPath = tmpPath
        pluginName = tmpPath.name
    pluginModule = _loadAModule(filePath = pluginPath,name = pluginName)
    pluginClass = pluginModule.__getattribute__("DLNestPlugin")
    return pluginClass

def getPluginConfig(pluginName, full = False):
    try:
        pluginClass : DLNestPluginBase = _loadAPlugin(pluginName)
        config = {}
        name = pluginClass.getName()
        if full:
            config[name] = pluginClass.getFullConfig()
        else:
            config[name] = pluginClass.getDefaultConfig()

        return name,config
    except Exception:
        return None, None