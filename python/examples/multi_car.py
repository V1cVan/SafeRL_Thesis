import pathlib
import time
import timeit
from hwsim import Simulation, BasicPolicy, KBModel, config
from hwsim.plotting import Plotter, SimulationPlot, DetailPlot, BirdsEyePlot, ActionsPlot

if __name__=="__main__":
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    FANCY_CARS = True
    ROOT = pathlib.Path(__file__).parent.absolute()
    SC_PATH = ROOT.joinpath("scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    #config.seed = 1249517370
    print(f"Using seed {config.seed}")
    sConfig = {
        "scenario": "CIRCUIT",
        "MAX_IT": 1000,
        "input_log": "",
        "k0": 0,
        "replay": False,
        "output_log": "" #f"logs/multi_car_{int(time.time())}.h5"
    }
    vTypes = [
        {"amount": 20, "model": KBModel(), "policy": BasicPolicy(BasicPolicy.Type.SLOW)},
        {"amount": 40, "model": KBModel(), "policy": BasicPolicy(BasicPolicy.Type.NORMAL)},
        {"amount": 7, "model": KBModel(), "policy": BasicPolicy(BasicPolicy.Type.FAST)}
    ]

    with Simulation(sConfig,vTypes) as sim:
        shape = (4,2)
        shape = (2,2)
        groups = [([0,1],0)]
        vehicle_type = "car" if FANCY_CARS else "box"
        p = Plotter(sim,"Multi car simulation",name="multi_car",mode=PLOT_MODE,shape=shape,groups=groups,off_screen=OFF_SCREEN)
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

        #print(timeit.timeit(lambda: sim.step(),number=100))
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