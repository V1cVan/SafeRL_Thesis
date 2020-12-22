import typing
import json
import yaml
import re

class Serializable(object):
    """ Base class for all Serializable classes. """
    yaml_flow_style = None
    decoder_unpack_mapping = True
    decoder_unpack_sequence = True

    def __init_subclass__(cls, enc_name=None, **kwargs):
        """ Subclasses are automatically registered if they provide a unique enc_name. """
        if enc_name is not None:
            cls.enc_name = enc_name
            BaseSerializer.register(cls.enc_name,cls)
            cls.yaml_tag = f"!{cls.enc_name}"
            YAMLSerializer.register(cls)
        super().__init_subclass__(**kwargs)

    def encode(self):
        """ Subclasses should return all extra data that has to be saved in order to
        fully restore this object later on (using decode). """
        return None # Uses default serialization of the subclass (see BaseSerializer.encode_serializable)

    @classmethod
    def decode(cls, data=None):
        """ If the object construction from the data dictionary is non-trivial, subclasses
        can override this class method. It should return an instance of the class, initialized
        from the data dictionary. """
        return None # Uses default construction of the subclass (see BaseSerializer.decode_serializable)

    def s_clone(self):
        """ Returns a clone of this object by chaining the decode and encode methods. """
        serialized = BaseSerializer.encode(self)
        return BaseSerializer.decode(serialized)

    @classmethod
    def to_yaml(cls, dumper, obj):
        data = BaseSerializer.encode_serializable(obj)
        if data is None:
            data = '~'
        if isinstance(data, typing.Mapping):
            node = dumper.represent_mapping(cls.yaml_tag, data, flow_style=cls.yaml_flow_style)
        elif isinstance(data, (typing.List, typing.Tuple)):
            node = dumper.represent_sequence(cls.yaml_tag, data, flow_style=cls.yaml_flow_style)
        else:
            node = dumper.represent_scalar(cls.yaml_tag, data)
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        if isinstance(node, yaml.MappingNode):
            data = loader.construct_mapping(node, deep=True)
        elif isinstance(node, yaml.SequenceNode):
            data = loader.construct_sequence(node, deep=True)
        else:
            data = loader.construct_scalar(node)
            if data=="~":
                data = None
        return BaseSerializer.decode_serializable(cls.enc_name, data)


class BaseSerializer(object):
    __classes = {}

    @classmethod
    def register(cls, name, S):
        assert name not in cls.__classes and issubclass(S,Serializable)
        cls.__classes[name] = S

    @classmethod
    def encode(cls, obj):
        if isinstance(obj, typing.Mapping):
            return {key: cls.encode(value) for key,value in obj.items()}
        elif isinstance(obj, (typing.List, typing.Tuple)):
            return [cls.encode(el) for el in obj]
        elif isinstance(obj, Serializable):
            data = cls.encode_serializable(obj)
            if data is None:
                return obj.enc_name # Default serialization does not save extra data, just the enc_name
            else:
                return {obj.enc_name: cls.encode(data)} # Otherwise a dictionary containing the extra data is saved
        elif isinstance(obj, (int,float,str)) or obj is None:
            return obj
        else:
            raise TypeError(f"Objects of type {type(obj)} are not supported by BaseSerializer.")

    @classmethod
    def decode(cls, data):
        if isinstance(data, typing.Mapping):
            for key,value in data.items():
                data[key] = cls.decode(value)
        elif isinstance(data, (typing.List, typing.Tuple)):
            data = [cls.decode(el) for el in data]

        # Check if data encodes a Serializable object:
        obj = None
        if isinstance(data, str) and data in cls.__classes:
            # Create Serializable without extra data
            obj = BaseSerializer.decode_serializable(data)
        elif isinstance(data, typing.Mapping) and len(data)==1:
            enc_name, enc_data = next(iter(data.items()))
            if enc_name in cls.__classes:
                # Create Serializable with extra data
                obj = BaseSerializer.decode_serializable(enc_name, enc_data)

        if obj is not None:
            return obj
        else:
            return data

    @classmethod
    def encode_serializable(cls, obj):
        data = obj.encode()
        if isinstance(data, typing.Sized) and len(data)==0:
            data = None
        return data

    @classmethod
    def decode_serializable(cls, enc_name, data=None):
        S = cls.__classes[enc_name]
        if data is None:
            return S.decode() or S() # First try decode method. If that returns None, use the default constructor
        obj = S.decode(data) # First try decode method.
        if obj is None: # If that returns None, call an appropriate constructor
            if isinstance(data, typing.Mapping) and S.decoder_unpack_mapping:
                return S(**data)
            elif isinstance(data, (typing.List, typing.Tuple)) and S.decoder_unpack_sequence:
                return S(*data)
            else:
                return S(data)
        else:
            return obj

class YAMLSerializer(object):

    @classmethod
    def register(cls, S):
        # Same magic as in YAMLObjectMetaclass:
        for loader in [yaml.Loader,yaml.SafeLoader,yaml.FullLoader]:
            loader.add_constructor(S.yaml_tag, S.from_yaml)
        for dumper in [yaml.Dumper,yaml.SafeDumper]:
            dumper.add_representer(S, S.to_yaml)


# Fix implicit resolution of floats in scientific notation (taken from https://stackoverflow.com/a/30462009/1176420)
for loader in [yaml.Loader,yaml.SafeLoader,yaml.FullLoader]:
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )


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
