import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorboard as tb
from tensorboard.backend.event_processing import event_multiplexer

"""
For uploading experiment to Tensorboard:
    tensorboard dev upload --logdir=="./logfiles/DDQN_ER_initialisers \
        --name "Name of the experiment" \
        --description "Description of the experiment"

For listing all the experiments you have done: 
    tensorboard dev list

For deleting experiment from Tensorboard:
    tensorboard dev delete --experiment_id YOUR_EXPERIMENT_ID_HERE
"""

if __name__ == "__main__":
    experiment_ids = {
        "DDQN_ER_initialisers": "8viGslanQLWtMaORGizRaQ"
    }
    experiment_paths = {
        "DDQN_ER_initialisers": "./logfiles/DDQN_ER_initialisers"
    }

    experiment_names = list(experiment_ids.keys())

    experiment_id = experiment_ids["DDQN_ER_initialisers"]
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    results = experiment.get_scalars(pivot=True)


    csv_path = experiment_paths["DDQN_ER_initialisers"] + "./" + experiment_names[0] + '.csv'
    results.to_csv(csv_path, index=False)
    results_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(results_df, results)
    print("Keys for dataframe: " + results_df.keys())

    td_errors_df = results_df[["run", "step", "Episode TD errors (sum)"]]
    losses_df = results_df[["run", "step", "Episode losses (sum)"]]
    batch_reward_df = results_df[["run", "step", "Episode mean batch rewards (sum)"]]
    mean_reward_df = results_df[["run", "step", "Mean episode reward"]]
    vehicle_speed_df = results_df[["run", "step", "Mean vehicle speed for episode"]]
    train_time_df = results_df[["run", "step", "Total time taken for episode"]]
    epsilon_df = results_df[["run", "step", "epsilon"]]

    run_names = results_df["run"].unique().tolist()
    for run_name in run_names:
        td_errors_df.loc(td_errors_df.run==run_name)

    grouped_results = list(results_df.groupby(["run"]))[3]

    print(results_df)


