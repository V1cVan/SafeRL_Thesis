from hwsim.serialization import Serializable

class Metric(Serializable):
    fields = [] # Should list all possible keys returned in the dictionary by the fetch method

    def init_vehicle(self, veh):
        """
        Subclasses can initialize some vehicle dependent properties here.
        """
        pass

    def fetch(self, veh):
        """
        Subclasses can implement this 'fetch' method, returning a dictionary
        of metrics that will be attached to the vehicle (accessible through
        veh.metrics).
        """
        return {}


# Support functional metric creation:
def metric(name, fields, fetch, init_vehicle=None):
    cls_body = {
        "fields": fields,
        "fetch": fetch
    }
    if init_vehicle is not None:
        cls_body["init_vehicle"] = init_vehicle

    return type(f"{name}Metric", (Metric,), cls_body, enc_name=name)
