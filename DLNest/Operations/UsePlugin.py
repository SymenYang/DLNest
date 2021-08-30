from DLNest.Operations.GetPluginConfig import getPluginConfig
from pathlib import Path
import json

def usePlugin(targetDir : str, pluginName : str, full : bool = False):
    """
    Only return False when targetDir is wrong
    """
    name, config = getPluginConfig(pluginName, full = full)
    if not config:
        print("Wrong plugin name: {}".format(pluginName))
        return True
    
    root_dir = Path(targetDir).absolute()
    
    if not root_dir.exists():
        print("Wrong target dir: {}".format(targetDir))
        return False
    
    plugins_config_path = root_dir / "plugins_config.json"
    if plugins_config_path.exists():
        with plugins_config_path.open("r") as f:
            pluginsConfig = json.load(f)
        if name in pluginsConfig["plugins"]:
            return True
    
        pluginsConfig["plugins"].append(name)
        pluginsConfig["plugins_config"].update(config)
        with plugins_config_path.open("w") as f:
            json.dump(pluginsConfig, f, indent = 4,separators = (',',':'))
    else:
        root_config_path = root_dir / "root_config.json"

        if not root_config_path.exists():
            print("Wrong target dir: {}".format(targetDir))
            return False

        with root_config_path.open("r") as f:
            allConfig = json.load(f)
        if name in allConfig["plugins"]:
            return True
        allConfig["plugins"].append(name)
        allConfig["plugins_config"].update(config)
        with root_config_path.open("w") as f:
            json.dump(allConfig, f, indent = 4,separators = (',',':'))
        
    return True

def usePlugins(targetDir, pluginsName : list, full : bool = False):
    for name in pluginsName:
        if not usePlugin(targetDir, name, full):
            return