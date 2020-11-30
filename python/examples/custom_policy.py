import pathlib
import random
import numpy as np
from hwsim import Simulation, BasicPolicy, IMPolicy, KBModel, ActionType, CustomPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, TimeChartPlot, ActionsPlot


class FixedLanePolicy(CustomPolicy, enc_name="fixed_lane"):
    """ Simple policy where each vehicle will stay in its initial lane with a certain target
    velocity (relative to the maximum allowed speed). The actual velocity is always upper
    bounded by the safety bounds (taking vehicles in front into account)."""
    LONG_ACTION = ActionType.ABS_VEL
    LAT_ACTION = ActionType.REL_OFF # Alternatively: ActionType.LANE
    
    def __init__(self):
        super().__init__()
        self.STEP_TIME = 100 # Change reference velocity every 100 iterations (10s)

    def init_vehicle(self, veh):
        """ Policy objects are shared over many different vehicles so to associate
        attributes to specific vehicles, we can use this method (which is called
        during Vehicle instantiation) """
        veh.rel_vel = 0
        veh.counter = 0

    def _set_rel_vel(self, veh):
        veh.rel_vel = 0.95-random.random()*0.3

    def custom_action(self, veh):
        """ This method is called at every iteration and the returned numpy arrary
        will be used as the new reference actions (passed to the lower level controllers
        who will set up proper model inputs to track the new reference) """
        # Start with updating the counter and setting a new reference if necessary
        veh.counter -= 1
        if veh.counter<=0:
            veh.counter = self.STEP_TIME
            self._set_rel_vel(veh)
        # Then calculate proper actions from the current reference
        s = veh.s # Current augmented state
        bounds = veh.a_bounds # Current safety bounds on the actions (calculated from the current augmented state). Vehicle operation remains 'safe' as long as we respect these bounds.
        v_max = veh.rel_vel*(s["maxVel"])
        v = min(v_max,bounds["long"][1])
        v = max(0,v)
        # Final actions are: the target velocity and negating the offset towards the lane center
        return np.array([v,-s["laneC"]["off"]])
        # Alternatively (with LANE actionType):
        # return np.array([v,0]) # Lane reference is 0 => remain in (center of) current lane


def simulate(sim):
    # Runs the given simulation and plots it
    with sim:
        shape = (4,2)
        groups = [(np.s_[:],0)]
        p = Plotter(sim,"Fixed lane simulation",mode=Plotter.Mode.LIVE,shape=shape,groups=groups)
        p.subplot(0,0)
        p.add_text("Detail view")
        DetailPlot(p,show_ids=True)
        p.add_overlay()
        SimulationPlot(p,vehicle_type=None,show_marker=True)
        p.subplot(0,1)
        p.add_text("Front view")
        BirdsEyePlot(p,vehicle_type="car",view=BirdsEyePlot.View.FRONT)
        p.subplot(1,1)
        p.add_text("Actions")
        lines = {
            "rel_vel": {
                "color": [0, 0, 0],
                "getValue": lambda veh: veh.rel_vel
            }
        }
        TimeChartPlot(p, lines, None, "rel_vel", [0])
        p.subplot(2,1)
        ActionsPlot(p,actions="long")
        p.subplot(3,1)
        ActionsPlot(p,actions="lat")
        p.plot() # Initial plot

        while not sim.stopped and not p.closed:
            sim.step()
            p.plot()

        p.close() # Make sure everything is closed correctly (e.g. video file is closed properly)


if __name__=="__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    config.seed = 1000 # For reproducability
    print(f"Using seed {config.seed}")
    # Outside a with-block, the Simulation object is just a configuration object that can be used
    # to (re)set certain parameters in between different runs. It is only within a with-block that
    # the Simulation object is actually backed by a C++ object and can be used to actually simulate
    # a run.
    sim = Simulation({"name": "fixed_lane", "scenario": "CIRCULAR", "kM": 1000})
    # Add 2 vehicles to the simulation, both with our fixed_lane custom policy.
    # The first vehicle is positioned at the start of the road (s=l=0), whereas
    # the second vehicle is slightly ahead (100 meters) in the rightmost lane.
    kbm = KBModel()
    custom_policy = FixedLanePolicy()
    sim.add_vehicles([
        {"model": kbm, "policy": custom_policy, "R": 0, "l": 0, "s": 0, "v": 30},
        {"model": kbm, "policy": custom_policy, "R": 0, "l": -3.6, "s": 100}
    ])
    # Perform actual simulation.
    # Note that during live plotting, you can press the following keys:
    #   * q to quit (abort simulation and close plot window)
    #   * p to pause/resume
    #   * s to step (one iteration is performed and afterwards back to pause)
    #   * v to change the focussed vehicle (equivalent of calling p.V=v_id in code)
    simulate(sim) # Perform actual simulation
    # After the simulation (outside the with-block), our sim object is once again a configuration
    # object that we can further modify for a subsequent run. In this case we will add two more
    # vehicles in the leftmost lane:
    sim.add_vehicles([
        {"model": kbm, "policy": IMPolicy(), "R": 0, "l": 3.6, "s": 25},
        {"model": kbm, "policy": BasicPolicy("slow"), "R": 0, "l": 3.6, "s": 50}
    ])
    simulate(sim)
