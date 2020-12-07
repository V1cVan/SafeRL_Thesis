from hwsim._wrapper import Blueprint

class _Model(Blueprint):

    def __init__(self, id, args=None):
        super().__init__(id,args)

    def custom_state(self, veh):
        """
        Subclasses can implement this 'custom_state' method to provide a custom
        updated state to the vehicle, overriding the default update from the base
        model in the next simulation step. The current input can be queried through
        the 'u_raw' or 'u' properties of the vehicle object. The default state
        vector, as calculated by the base model, can also be queried through the
        'x_raw' or 'x' properties.
        """
        # TODO: currently not implemented in Simulation
        return None

    def custom_derivatives(self, veh, x, u):
        """
        Subclasses can implement this 'custom_derivatives' method to override the
        default derivatives, used to calculate an updated state vector, from the base
        model in the next simulation step. The current input can be queried through
        the 'u_raw' or 'u' properties of the vehicle object. The default derivatives,
        as calculated by the base model, can also be queried through the 'dx_raw' or
        'dx' properties.
        """
        # TODO: currently not implemented in Simulation and Vehicle
        return None

    def custom_nominal_input(self, veh, x, gamma):
        """
        Subclasses can implement this 'custom_nominal_input' method to override
        the default nominal inputs of the base model. If a 'custom_derivatives'
        method is defined, this method will have to be implemented as well, to
        guarantee a normal steady-state behaviour of the vehicle (following lanes).
        """
        # TODO: currently not implemented in Simulation and Vehicle
        return None


class KBModel(_Model,enc_name="kbm"):
    """
    Kinematic bicycle model
    """

    def __init__(self):
        super().__init__(1)


class DBModel(_Model,enc_name="dbm"):
    """
    Dynamic bicycle model
    """

    def __init__(self):
        super().__init__(2)


class CustomModel(_Model):

    def __init__(self):
        super().__init__(0)
