from functools import wraps
import logging

def checkPlugins(func):
    @wraps(func)
    def checkAndRun(*args, **kwargs):
        name = func.__name__
        for plugin in args[0]._plugins:
            if name[1:] in dir(plugin):
                try:
                    getattr(plugin,name[1:])(*args, **kwargs)
                except Exception as e:
                    logging.debug(str(e))
        return func(*args, **kwargs)
    
    return checkAndRun

def checkDictOutputPlugins(func):
    @wraps(func)
    def checkAndRun(*args, **kwargs):
        name = func.__name__
        ret = {}
        for plugin in args[0]._plugins:
            if name[1:] in dir(plugin):
                try:
                    ret.update(getattr(plugin,name[1:])(*args, **kwargs))
                except Exception as e:
                    logging.debug(str(e))
        ret.update(func(*args, **kwargs))
        return ret
    
    return checkAndRun