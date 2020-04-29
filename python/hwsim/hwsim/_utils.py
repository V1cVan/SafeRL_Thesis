import numpy as np

def attr(name,attr):
    """
    Decorator that can be used to set an attribute on a class or method
    """
    def wrapper(func):
        setattr(func,name,attr)
        return func
    return wrapper