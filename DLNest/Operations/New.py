import shutil
from pathlib import Path
import json
from DLNest.Operations.UsePlugin import usePlugin

def new(targetDir : str, MNIST : bool = False, pluginsName = []):
    projectPath = Path(targetDir).absolute()
    # If target path already exists, return
    if projectPath.exists():
        print("Path already exists")
        return

    # Get the factory file
    factoryPath = Path(__file__).parent.parent / "FactoryFiles"
    if MNIST:
        factoryPath = factoryPath / "FactoryMNIST"
    else:
        factoryPath = factoryPath / "FactoryClean"
    
    # Copy to target
    shutil.copytree(factoryPath,projectPath)

    # Modify save_root and root_file_path in root_config.json
    rootConfigPath = projectPath / "root_config.json"
    rootConfig = {}
    with rootConfigPath.open("r") as fp:
        rootConfig = json.load(fp)
    rootConfig["save_root"] = str(projectPath / "Saves")
    rootConfig["root_file_path"] = str(projectPath)
    with rootConfigPath.open("w") as fp:
        json.dump(rootConfig,fp,indent = 4,separators = (',',':'))
    
    # If MNIST, modify the data_root in dataset_config.json
    if MNIST:
        datasetConfigPath = projectPath / "dataset_config.json"
        datasetConfig = {}
        with datasetConfigPath.open("r") as fp:
            datasetConfig = json.load(fp)
        datasetConfig["dataset_config"]["data_root"] = str(projectPath / "MNIST")
        with datasetConfigPath.open("w") as fp:
            json.dump(datasetConfig,fp,indent = 4, separators = (',',':'))
        
    
    # Add plugins
    for pluginName in pluginsName:
        usePlugin(targetDir, pluginName = pluginName, full = False)

    print("Create a project in " + targetDir + ".")
