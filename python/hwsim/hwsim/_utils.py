import random
import numpy as np
from hwsim._wrapper import simLib


class hybridmethod:
    """
    This decorator allows a class to have a static method, class method, instance method
    and/or property with the same name. Note that priority will always be given to class or
    instance methods. Hence when both are provided, the static method will never be called.
    Similarly a provided instance method will override the instance property.
    This is a slightly altered version of https://stackoverflow.com/a/28238047
    A good explanation of the property object (which is both a decorator and descriptor):
    https://stackoverflow.com/a/17330273
    """
    def __init__(self, fstatic, fclass=None, finst=None, finst_getter=None, finst_setter=None, finst_deleter=None, doc=None):
        self.fstatic = fstatic
        self.fclass = fclass
        self.finst = finst
        self.__doc__ = doc or fstatic.__doc__
        self.prop = property(finst_getter,finst_setter,finst_deleter,self.__doc__)

    def staticmethod(self, fstatic):
        return type(self)(fstatic, self.fclass, self.finst, self.prop.fget, self.prop.fset, self.prop.fdel, self.__doc__)

    def classmethod(self, fclass):
        return type(self)(self.fstatic, fclass, self.finst, self.prop.fget, self.prop.fset, self.prop.fdel, self.__doc__)

    def instancemethod(self, finst):
        return type(self)(self.fstatic, self.fclass, finst, self.prop.fget, self.prop.fset, self.prop.fdel, self.__doc__)

    def instancegetter(self, finst_getter):
        return type(self)(self.fstatic, self.fclass, self.finst, finst_getter, self.prop.fset, self.prop.fdel, self.__doc__)

    def instancesetter(self, finst_setter):
        return type(self)(self.fstatic, self.fclass, self.finst, self.prop.fget, finst_setter, self.prop.fdel, self.__doc__)

    def instancedeleter(self, finst_deleter):
        return type(self)(self.fstatic, self.fclass, self.finst, self.prop.fget, self.prop.fset, finst_deleter, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None:
            if self.fclass is not None:
                # Class method
                return self.fclass.__get__(cls, None)
        else:
            if self.finst is not None:
                # Instance method
                return self.finst.__get__(instance, cls)
            else:
                # Property
                return self.prop.__get__(instance, cls)
        # Otherwise call the static method:
        return self.fstatic
    
    def __set__(self, instance, value):
        # TODO: if we ever need this, it might be easier to just inherit from property?
        self.prop.__set__(instance, value)
    
    def __delete__(self, instance):
        self.prop.__delete__(self, instance)


class Configuration(object):

    def __init__(self):
        self._seed = simLib.cfg_getSeed()
        self.__set_python_seed(self._seed)
        self.scenarios_path = "scenarios.h5"
    
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self,newSeed):
        self._seed = newSeed
        self.__set_python_seed(self._seed)
        simLib.cfg_setSeed(newSeed)
    
    @staticmethod
    def __set_python_seed(newSeed):
        random.seed(newSeed)
        # TODO: set numpy seed?
    
    @property
    def scenarios_path(self):
        return self._scenarios_path

    @scenarios_path.setter
    def scenarios_path(self,path):
        self._scenarios_path = path
        simLib.cfg_scenariosPath(path.encode("utf8"))

config = Configuration()