from functools import wraps


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


class conditional(object):
    """
    This decorator allows a method to be toggled depending on a certain condition
    on its parameters. If the condition holds, the method is executed for the given
    parameters. Otherwise a RuntimeError is thrown.
    """
    def __init__(self, cond):
        # Creates the decorator
        self.fcond = cond

    def __call__(self, func):
        # Applies the decorator
        @wraps(func)
        def wrapper(*f_args,**f_kwargs):
            if self.fcond(*f_args,**f_kwargs):
                return func(*f_args,**f_kwargs)
            else:
                raise RuntimeError(f"Condition was not satisfied. {self.fcond} returned False for arguments {f_args},{f_kwargs}")
        return wrapper

    @staticmethod
    def NOT(cond):
        return lambda *f_args,**f_kwargs: not cond(*f_args,**f_kwargs)

    @staticmethod
    def OR(cond1,cond2):
        return lambda *f_args,**f_kwargs: cond1(*f_args,**f_kwargs) or cond2(*f_args,**f_kwargs)

    @staticmethod
    def AND(cond1,cond2):
        return lambda *f_args,**f_kwargs: cond1(*f_args,**f_kwargs) and cond2(*f_args,**f_kwargs)
