import typing
import json
# import re

class Serializable(object):
    """ Base class for all Serializable classes. """

    def __init_subclass__(cls, enc_name=None, **kwargs):
        """ Subclasses are automatically registered if they provide a unique enc_name. """
        if enc_name is not None:
            cls.enc_name = enc_name
            BaseSerializer.register(enc_name,cls)
        super().__init_subclass__(**kwargs)

    def encode(self):
        """ Subclasses should return all extra data that has to be saved in order to
        fully restore this object later on (using decode). """
        return None # Uses default serialization of the subclass (see BaseSerializer.encode_serializable)

    @staticmethod
    def decode(data=None):
        """ If the object construction from the data dictionary is non-trivial, subclasses
        can override this static method. It should return an instance of the class, initialized
        from the data dictionary. """
        return None # Uses default construction of the subclass (see BaseSerializer.decode_serializable)

    def s_clone(self):
        """ Returns a clone of this object by chaining the decode and encode methods. """
        serialized = BaseSerializer.encode(self)
        return BaseSerializer.decode(serialized)


class BaseSerializer(object):
    __classes = {}

    @classmethod
    def register(cls, name, S):
        assert name not in cls.__classes and issubclass(S,Serializable)
        cls.__classes[name] = S

    @classmethod
    def encode(cls, obj):
        if isinstance(obj,typing.Mapping):
            for key,value in obj.items():
                obj[key] = cls.encode(value)
            return obj
        elif isinstance(obj,typing.List):
            return [cls.encode(el) for el in obj]
        elif isinstance(obj,Serializable):
            return cls.encode_serializable(obj)
        elif isinstance(obj,(int,float,str)):
            return obj
        else:
            raise TypeError(f"Objects of type {type(obj)} are not supported by BaseSerializer.")

    @classmethod
    def decode(cls, data):
        if isinstance(data,typing.Mapping):
            for key,value in data.items():
                data[key] = cls.decode(value)
        elif isinstance(data,typing.List):
            data = [cls.decode(el) for el in data]
        obj = cls.decode_serializable(data)
        if obj is not None:
            return obj
        else:
            return data

    @staticmethod
    def encode_serializable(obj):
        data = obj.encode()
        if data is None or (isinstance(data,typing.Sized) and len(data)==0):
            return obj.enc_name # Default serialization does not save extra data, just the enc_name
        return {obj.enc_name: data} # Otherwise a dictionary containing the extra data is saved

    @classmethod
    def decode_serializable(cls, data):
        if isinstance(data,str) and data in cls.__classes:
            S = cls.__classes[data]
            # Create without extra data
            return S.decode() or S() # First try decode method. If that returns None, use the default constructor
        elif isinstance(data,typing.Dict) and len(data)==1:
            name = next(iter(data))
            if name in cls.__classes:
                S = cls.__classes[name]
                # Create with extra data
                obj = S.decode(data[name]) # First try decode method.
                if obj is None: # If that returns None, call an appropriate constructor
                    if isinstance(data[name],typing.Dict):
                        return S(**data[name])
                    elif isinstance(data[name],typing.List):
                        return S(*data[name])
                    else:
                        return S(data[name])
                else:
                    return obj
        return None


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Serializable):
            return BaseSerializer.encode_serializable(obj)
        return super().default(obj)

    # def encode(self, o):
    #     result = super().encode(o)

    #     def inner_list_replace(match):
    #         # Deletes all whitespace from the inner list match
    #         return re.sub(r"\s+","",match.string)

    #     # Matches all innermost lists (lists not containing other lists or dicts)
    #     # Match '[', followed by any tokens not in the set {'[',']','{','}'},
    #     # followed by ']'
    #     return re.sub(r"\[[^\[\]\{\}]+\]",inner_list_replace,result)

    def iterencode(self, o, *args, **kwargs):
        """ This method wraps the iterencode generator of `json.JSONEncoder` and
        analyses the tokens produced by this super iterator. It assumes complex
        (nested) lists produce a single '[\s' token , whereas simple (inner) lists
        produce a token containing '[\s' together with the first element. Every
        subsequent element produces a token containing ',\s' together with the
        element. Finally, the end of any list is denoted by a single ']' token.
        It uses these assumptions to remove the whitespace from inner lists. """
        super_iterator = super().iterencode(o,*args,**kwargs)
        item_sep = self.item_separator

        def generator():
            inner_list = False
            for token in super_iterator:
                if "[" in token and token.strip()!="[":
                    inner_list = True
                elif token=="]":
                    inner_list = False
                if inner_list:
                    if "[" in token:
                        token = "[" + token[1:].strip()
                    elif item_sep in token:
                        token = item_sep + token[len(item_sep):].strip()
                    else:
                        continue # Skip '\s' token right before end of the list
                yield token
        return generator()


class JSONDecoder(json.JSONDecoder):
    def __init__(self,*args,**kwargs):
        super().__init__(object_hook=self.object_hook,*args,**kwargs)

    def object_hook(self, dct):
        # We iterate over the values because simple serializable classes are
        # encoded as a string, which is not passed to this hook.
        for key,val in dct.items():
            obj = BaseSerializer.decode_serializable(val)
            if obj is not None:
                dct[key] = obj
        return dct
