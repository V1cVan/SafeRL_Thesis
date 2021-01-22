import pathlib
import timeit
from hwsim import Simulation, BasicPolicy, IMPolicy, StepPolicy, SwayPolicy, KBModel, config
from hwsim.plotting import Plotter, BirdsEyePlot, ActionsPlot

# Buckle up, because this will be quite the ride...
# Used to verify safety of the action bounds.
if __name__=="__main__":
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")
    # Safety configuration and N_OV of SwayPolicy:
    N_OV = 2
    safetyCfg = {
        "Mvel": 1.0,
        "Gth": 2.0
    }

    config.scenarios_path = str(SC_PATH)
    # config.seed = 752672697
    print(f"Using seed {config.seed}")
    output_dir = ""
    vTypes = [
        {"amount": 2, "model": KBModel(), "policy": StepPolicy(10,[0.1,0.5])},
        {"amount": 1, "model": KBModel(), "policy": SwayPolicy(), "N_OV": N_OV, "safety": safetyCfg},
        {"amount": 8, "model": KBModel(), "policy": IMPolicy()},
        {"amount": 3, "model": KBModel(), "policy": BasicPolicy("slow")},
        {"amount": 25, "model": KBModel(), "policy": BasicPolicy("normal")},
        {"amount": 7, "model": KBModel(), "policy": BasicPolicy("fast")}
    ]
    sConfig = {
        "name": f"stress_test",
        "scenario": "CIRCUIT",
        "kM": 5000,
        "output_dir": output_dir,
        "vehicles": vTypes
    }

    with Simulation(sConfig) as sim:
        shape = (2,2)
        p = Plotter(sim,"Stress test",mode=PLOT_MODE,shape=shape,off_screen=OFF_SCREEN)
        p.V = 2
        p.subplot(0,0)
        p.add_text("Front view")
        BirdsEyePlot(p,vehicle_type="car",view=BirdsEyePlot.View.FRONT)
        p.subplot(1,0)
        p.add_text("Rear view")
        BirdsEyePlot(p,vehicle_type="car",view=BirdsEyePlot.View.REAR)
        p.subplot(0,1)
        p.add_text("Actions")
        ActionsPlot(p,actions="long")
        p.subplot(1,1)
        ActionsPlot(p,actions="lat")
        p.plot() # Initial plot

        while not sim.stopped and not p.closed:
            start = timeit.default_timer()
            sim.step()
            end = timeit.default_timer()
            print(f"Simulation step took {(end-start)*1000} ms")
            start = timeit.default_timer()
            p.plot()
            end = timeit.default_timer()
            print(f"Drawing took {(end-start)*1000} ms")

        p.close() # Make sure everything is closed correctly (e.g. video file is closed properly)
