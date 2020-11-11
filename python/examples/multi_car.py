import pathlib
import time
import timeit
from hwsim import Simulation, BasicPolicy, KBModel, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, ActionsPlot

if __name__=="__main__":
    ID = -1 # ID of simulation to replay or -1 to create a new one
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    FANCY_CARS = True
    LOG_DIR = "logs"
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    SC_PATH = ROOT.joinpath("scenarios/scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    # config.seed = 1249517370
    print(f"Using seed {config.seed}")
    if ID<0:
        ID = int(time.time())
        print(f"Creating new multi car simulation with ID {ID}")
        input_dir = ""
        output_dir = LOG_DIR
        vTypes = [
            {"amount": 20, "model": KBModel(), "policy": BasicPolicy("slow")},
            {"amount": 40, "model": KBModel(), "policy": BasicPolicy("normal")},
            {"amount": 7, "model": KBModel(), "policy": BasicPolicy("fast")}
        ]
    else:
        print(f"Replaying multi car simulation with ID {ID}")
        input_dir = LOG_DIR
        output_dir = ""
        vTypes = []
    sConfig = {
        "name": f"multi_car_{ID}",
        "scenario": "CIRCUIT",
        "kM": 1000,
        "input_dir": input_dir,
        "k0": 0,
        "replay": True,
        "output_dir": output_dir,
        "vehicles": vTypes
    }

    with Simulation(sConfig) as sim:
        shape = (4,2)
        shape = (2,2)
        groups = [([0,1],0)]
        vehicle_type = "car" if FANCY_CARS else "cuboid3D"
        p = Plotter(sim,"Multi car simulation",mode=PLOT_MODE,shape=shape,groups=groups,off_screen=OFF_SCREEN)
        p.V = 64
        p.subplot(0,0)
        p.add_text("Detail view")
        DetailPlot(p,show_ids=True)
        p.add_overlay()
        SimulationPlot(p,vehicle_type=None,show_marker=True)
        p.subplot(0,1)
        p.add_text("Front view")
        BirdsEyePlot(p,vehicle_type=vehicle_type,view=BirdsEyePlot.View.FRONT)
        p.subplot(1,1)
        p.add_text("Rear view")
        BirdsEyePlot(p,vehicle_type=vehicle_type,view=BirdsEyePlot.View.REAR)
        # p.subplot(2,0)
        # p.add_text("Actions")
        # ActionsPlot(p,actions="vel")
        # p.subplot(3,0)
        # ActionsPlot(p,actions="off")
        # p.subplot(2,1)
        # p.subplot(3,1)
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
