class ActorNet(tf.keras.models.Model):
    """ Definition of the network architecture for the actor networks. """

    def __init__(self, state_dim=S, action_dim=A):
        # Initialize keras fully connected layers
        self.dense1 = tf.keras.layers.Dense(H, input_dim=state_dim, ...)

    # ...

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


class CriticNet(tf.keras.models.Model):
    """ Definition of the network architecture for the critic networks. """


# Similar to ActorNet


class DDPG_Trainer(tf.keras.models.Model):
    """ Helper object to guide the training process of the networks. Will depend on your chosen RL method. """

    def __init__(self, actor, critic, cfg):
        self.actor = actor
        self.actorT = actor.clone()  # I wrote a custom clone method to easily clone the network architecture and weights
        self.critic = critic
        self.criticT = critic.clone()
        self.cfg = cfg
        self.buffer = Buffer()  # Custom buffer object, could be a simple python deque

    def add_experience(self, s0, a, r, s1):
        if self.training:
            self.buffer.store(s0, a, r, s1)

    def train_step(self):
        print()


# Update the actor and critic weights, will depend on the chosen RL method.
# E.g. fetch random experience sample from buffer, calculate critic loss, update critic,
# calculate actor loss, update actor, update target networks


class AutoPolicy(CustomPolicy):
    def __init__(self, trainer):
        self.trainer = trainer

    def init_vehicle(self, veh):
        # Bookkeeping of last states and actions
        veh.s0 = None  # Previous vehicle state

    veh.s0_mod = None  # Previous vehicle state as passed to the actor and critic models
    veh.s1 = None  # Current vehicle state
    veh.s1_mod = None  # Current vehicle state as passed to the actor and critic models
    veh.a0 = None
    veh.a0_mod = None
    veh.a1 = None
    veh.a1_mod = None
    veh.metrics = self.get_metrics()


def custom_action(self, veh):
    # Update bookkeeping
    veh.s0 = veh.s1
    veh.s0_mod = veh.s1_mod
    veh.a0 = veh.a1
    veh.a0_mod = veh.a1_mod
    veh.s1 = veh.s
    veh.s1_mod = self.convert_state(veh).astype(np.float32)  # Our networks use single precision floats
    # And get new actions based on new states
    veh.a1_mod = self.get_action(veh)
    veh.a1 = self.convert_action(veh)
    if veh.s0_mod is not None:
        # Calculate new metrics
        veh.metrics = self.get_metrics(veh)
        # And report new experience to trainer, if available
        if self.trainer is not None:
            self.trainer.add_experience(veh.s0_mod, veh.a0_mod, veh.metrics["reward"], veh.s1_mod)
    return np.array([veh.a1]).view(np.float64)  # The hwsim library uses double precision floats

    def convert_state(self, veh):
        """ Get the modified state vector that will be passed to the actor and critic models from the
        state vector (available in veh). I.e. the mapping s->s_mod """

    return veh.s_raw[:S]  # Can be overriden by subclasses


def convert_action(self, veh):
    """ Get the action vector that will be passed to the vehicle from the given model action vector
    (used by the actor and critic models and available in veh). I.e. the mapping a_mod->a """
    return veh.a1_mod  # Can be overriden by subclasses


def get_action(self, veh):
    """ Get the modified action vector from the modified state vector. I.e. the mapping s_mod->a_mod """
    return self.trainer.actor(veh.s1_mod.reshape((1, -1))).numpy().flatten()


def get_metrics(self, veh=None):
    # Calculate the reward and any other metrics you want to keep track of for this policy
    return {"reward": 0}  # Can be overriden by subclasses


class Main(object):

    def __init__(self, params):
        # ...
        self.sim = Simulation(params.sim_config)
        self.auto = [{  # List of autonomous policies
            "ids": [0],  # Vehicle IDs of vehicles equiped with this autonomous policy
            "policy": params.sim_config["vehicles"][0]["policy"],  # Reference to the policy
            "metrics": {}  # Extra metrics we calculate for each autonomous policy
        }]

    # ...

    def simulate(self):
        with self.sim:
            while not self.sim.stopped:
                # Perform one simulation step:
                self.sim.step()
                # ... (visualization/extra callbacks)

                # Keep track of average metrics in this episode
                # ...

                # Perform one train step, using the collected new experience:
                for auto in self.auto:
                    if auto["policy"].trainer.training:
                        auto["policy"].trainer.train_step()


if __name__ == "__main__":
    params = fetch_params()  # Reads parameters from configuration file (JSON or YAML) and extra command line arguments
    # params.sim_config is a dictionary containing all simulation configuration, such as the autonomous policy:
    # params.sim_config["vehicles"][0] = AutoPolicy(DDPG_Trainer(ActorNet(),CriticNet()))
    Main(params).run()  # Run will start multiple simulations and call 'simulate()' on each of them