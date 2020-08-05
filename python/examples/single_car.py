import pathlib
import timeit
from hwsim import Simulation, KBModel, StepPolicy, BasicPolicy, config
from hwsim.plotting import Plotter, SimulationPlot, BirdsEyePlot, ActionsPlot, InputsPlot

if __name__=="__main__":
    PLOT_MODE = Plotter.Mode.LIVE
    OFF_SCREEN = False
    ROOT = pathlib.Path(__file__).parent.absolute()
    SC_PATH = ROOT.joinpath("scenarios.h5")

    config.scenarios_path = str(SC_PATH)
    print(f"Using seed {config.seed}")
    sConfig = {
        "scenario": "CLOVERLEAF",
        "MAX_IT": 1000
    }
    #policy = BasicPolicy(BasicPolicy.Type.NORMAL)
    policy = StepPolicy()
    vTypes = [
        {"amount": 1, "model": KBModel(), "policy": policy}
    ]

    with Simulation(sConfig,vTypes) as sim:
        shape = (4,2)
        groups = [([0, 1], 0), ([0, 1], 1)]
        p = Plotter(sim,"Single car simulation",name="single_car",mode=PLOT_MODE,shape=shape,groups=groups,off_screen=OFF_SCREEN)
        p.subplot(0,0)
        p.add_text("Scenario view")
        SimulationPlot(p,vehicle_type=None,show_marker=True)
        p.subplot(0,1)
        p.add_text("Bird's eye view")
        BirdsEyePlot(p,view=BirdsEyePlot.View.FRONT,vehicle_type="car")
        p.subplot(2,0)
        p.add_text("Actions")
        ActionsPlot(p,actions="vel")
        p.subplot(3,0)
        ActionsPlot(p,actions="off")
        p.subplot(2,1)
        p.add_text("Inputs")
        InputsPlot(p,inputs="acc")
        p.subplot(3,1)
        InputsPlot(p,inputs="delta")
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