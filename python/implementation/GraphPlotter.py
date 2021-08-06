import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard as tb
import numpy as np
import multiprocessing as mp
import PIL.Image as img

"""
For uploading experiment to Tensorboard:
    tensorboard dev upload 
        --logdir=="./logfiles/experiment/trainortest
        --name "Name of the experiment" 
        --description "Description of the experiment"

For listing all the experiments you have done: 
    tensorboard dev list

For deleting experiment from Tensorboard:
    tensorboard dev delete --experiment_id YOUR_EXPERIMENT_ID_HERE
"""

# tensorboard dev delete --experiment_id "nQWFNCwYSPWCJzy04EabMA"
# tensorboard dev upload --logdir "./logfiles/DDQN_ER_tuning/train" --name "DDQN ER Tuning run 2" --description "DDQN ER Tuning parameter sweep 2. Default var: Policy action rate=8; Units=(32,32); Activation=relu; Batch size=32; Learning rate=0.0001; Loss func=Huber; Clip grads=False; Gamma=0.95; Target update rate=1e4; Standardise returns=False"
# tensorboard dev upload --logdir "./logfiles/DDQN_Epsilon_Beta_tuning/train" --name "Epsilon tuning sweep" --description "DDQN Epsilon parameter sweep. Default var: Policy action rate=8; Units=(32,32); Activation=relu; Batch size=32; Learning rate=0.0001; Loss func=Huber; Clip grads=False; Gamma=0.95; Target update rate=1e4; Standardise returns=False; Buffer size=300000"
# tensorboard dev upload --logdir "./logfiles/DQN_DDQN_standardisation_target_update/train" --name "DQN with target standardisation vs DDQN" --description "Comparison with shorter learning times and less exploration. Default var: Policy action rate=8; Units=(32,32); Activation=relu; Batch size=32; Learning rate=0.0001; Loss func=Huber; Clip grads=False; Gamma=0.95; Target update rate=1e4; Standardise returns=False; Buffer size=300000"
# tensorboard dev upload --logdir "./logfiles/DQN_DDQN_standardisation_target_update_long_run/train" --name "DQN with target standardisation vs DDQN - long run" --description "Comparison with longer learning times and more exploration."

def start_run(parameter, tag_value, df, fig_path):

    sns.set_style("darkgrid")

    print(f"Plotting parameter={parameter}, tag value={tag_value}...")
    # Start your training loop using the given arguments
    plot_df = df.loc[(df['parameter_tested'] == parameter) & (df['tag'] == tag_value)]
    # plot_df = plot_df.iloc[::10,:] #Clearer by not plotting every step
    ax = sns.lineplot(x="step", y="smoothed",
                  hue="description",
                  data=plot_df,
                  alpha=0.6,
                  linewidth=1.8)
    if tag_value == "Epsilon":
        ax.get_legend().remove()
    else:
        ax.legend(loc='lower right', title='Parameter:')
    ax.set(xlabel = "Episode")
    ax.set(ylabel = tag_value)
    plt_name = fig_path + "/" + parameter + '-' + tag_value + '.png'
    plt.savefig(plt_name, dpi=300, bbox_inches='tight')
    print(f"Arg1: {parameter} ; Arg2: {tag_value}")
    plt.show()



def download_and_save(experiment_name, csv_path):
    # Retrieve the training data from Tensorboard Dev
    experiment_id = experiment_ids[experiment_name]
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    results = experiment.get_scalars()
    print("Download complete ... ")

    # Save downloaded data to a csv file
    results.to_csv(csv_path, index=False)
    results_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(results_df, results)
    print("Keys for dataframe: " + results_df.keys())
    print("Csv saved ... ")

# Function for extracting the regex patterns
def extract_columns(df, column1='seed', column2='description'):
    # Extract the seed to a column
    df[column1] = df['run'].str.extract(r'(Seed\d\d\d)')  # Seed must be three digits
    df[column1] = df[column1].str[4:]
    df[column1] = df[column1].astype('category')
    # Can access the different categories using shortened_df['seed'].dtypes.categories[index]

    # Extract the run description to a column
    df[column2] = df['run'].str.extract(r'(Details=.*)')
    df[column2] = df[column2].str[8:]
    return df


def preprocess_data(csv_path):
    # (Re) Load the dataset
    df = pd.read_csv(csv_path)
    assert df.shape is not None

    # Remove variables that are not Episode reward or Vehicle velocity
    df = extract_columns(df)
    # df = df.loc[df['tag'].isin(['Mean episode reward', 'Mean vehicle speed for episode'])]
    # df['parameter_tested'] = df['description'].str.extract(r'(.*?)-')
    df = df.loc[df['tag'].isin(['Reward', 'Vehicle speed', 'Epsilon'])]
    df['parameter_tested'] = df['description'].str.extract(r'(.*?)=')


    print(df.head())
    print(df["parameter_tested"].unique())

    # Smooth the results
    for i in df['description'].unique():  # Loop through parameters
        for j in df['seed'].dtypes.categories:  # Loop through different seeds
            for k in df['tag'].unique():  # Loop through Mean episode reward or mean vehicle speed
                df_selection = df.loc[(df['description'] == i) & (df['seed'] == j) & (
                        df['tag'] == k)]  # Select just the data that matches all 3 criteria
                raw_np = df_selection['value'].to_numpy()  # Convert pandas series to numpy array
                smooth_np = np.empty_like(raw_np)  # Initialize smooth_data_array
                # Initialise values
                smooth_np[0] = raw_np[0]  # First value in smoothed array is same as first value in raw
                if k == "Epsilon":
                    weight = 0
                else:
                    weight = 0.95
                for step in range(1, len(raw_np)):
                    smooth_np[step] = (1 - weight) * raw_np[step] + weight * smooth_np[step - 1]
                # Add np_array to df
                df.loc[(df['description'] == i) & (df['seed'] == j) & (df['tag'] == k), 'smoothed'] = smooth_np.tolist()

    print(f"Values smoothed by factor {weight}...")
    print(df.keys())
    print(df["smoothed"])

    return df


if __name__ == "__main__":
    experiment_ids = {
        "DDQN_ER_initialisers": "8viGslanQLWtMaORGizRaQ",
        "DDQN_ER_tuning": "ndalAuYXRxGuML7vkgPA4Q",
        "Epsilon_sweep": "t1GY8hx8QFWiNjyofqJUNw",
        "DQN_DDQN_standardisation_short": "0o0v4ZgrTWWUnOUlBrWmow",
        "DQN_DDQN_standardisation_long": "rhGbDukaTL28a8hjZO6QiQ",
    }

    experiment_paths = {
        "DDQN_ER_initialisers": "./logfiles/DDQN_ER_initialisers",
        "DDQN_ER_tuning": "./logfiles/DDQN_ER_tuning/train",
        "Epsilon_sweep": "./logfiles/DDQN_Epsilon_Beta_tuning/train",
        "DQN_DDQN_standardisation_short": "./logfiles/DQN_DDQN_standardisation_target_update/train",
        "DQN_DDQN_standardisation_long": "./logfiles/DQN_DDQN_standardisation_target_update_long_run/train",
    }

    experiment_names = list(experiment_ids.keys())
    print(experiment_names)

    experiment_name = "DQN_DDQN_standardisation_long"
    csv_path = experiment_paths[experiment_name] + "/" + experiment_name + '.csv'
    download_and_save(experiment_name, csv_path)  # Comment out if you don't want to re-download the data

    # Pre-process (smooth data from csv)
    df = preprocess_data(csv_path)

    # Plot the smoothed results using multi processing
    procs = 32  # Amount of processes/cores you want to use
    mp.set_start_method('spawn')  # This will make sure the different workers have different random seeds
    P = mp.cpu_count()  # Amount of available procs
    procs = max(min(procs, P), 1)  # Clip amount of procs to [1;P]

    def param_gen():
        # This function should return (yield because it's a generator) all parameter combinations you want to try
        for parameter in df['parameter_tested'].unique():
            for tag_value in df['tag'].unique():
                yield parameter, tag_value, df, experiment_paths[experiment_name]

    if procs > 1:
        # Schedule all training runs in a parallel loop when using multiple cores:
        # This does the same as the for loop in the else clause, but in parallel
        with mp.Pool(processes=procs) as pool:
            # Schedule all training runs in a parallel loop:
            pool.starmap(start_run, param_gen())
            pool.close()
            pool.join()
    else:
        # To make debugging easier when using 1 core, a classical for loop:
        for args in param_gen():
            start_run(*args)

    print("EOF")
