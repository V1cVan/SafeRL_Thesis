import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard as tb
import numpy as np
import multiprocessing as mp

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

def start_run(parameter, tag_value, df):
    sns.set_style("darkgrid")
    # Start your training loop using the given arguments
    plot_df = df.loc[(df['parameter_tested'] == parameter) & (df['tag'] == tag_value)]
    # plot_df = plot_df.iloc[::10,:] #Clearer by not plotting every step
    ax = sns.lineplot(x="step", y="smoothed",
                      hue="description",
                      data=plot_df,
                      alpha=0.6,
                      linewidth=1.5)
    ax.set_title(str(tag_value))
    print(f"Arg1: {parameter} ; Arg2: {tag_value}")
    plt.show()


def download_and_convert(experiment_name, csv_path):
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

if __name__ == "__main__":
    experiment_ids = {
        "DDQN_ER_initialisers": "8viGslanQLWtMaORGizRaQ",
        "DDQN_ER_tuning": "PWLK2gQmS0Wfm2c56YMjtA"
    }

    experiment_paths = {
        "DDQN_ER_initialisers": "./logfiles/DDQN_ER_initialisers",
        "DDQN_ER_tuning": "./logfiles/DDQN_ER_tuning/train"
    }

    experiment_names = list(experiment_ids.keys())
    print(experiment_names)

    experiment_name = "DDQN_ER_tuning"
    csv_path = experiment_paths[experiment_name] + "/" + experiment_names[1] + '.csv'
    # download_and_convert(experiment_name, csv_path)

    # (Re) Load the dataset
    df = pd.read_csv(csv_path)
    assert df.shape is not None

    # Remove variables that are not Episode reward or Vehicle velocity
    df = extract_columns(df)
    df = df.loc[df['tag'].isin(['Episode reward', 'Episode vehicle speed'])]
    df['parameter_tested'] = df['description'].str.extract(r'(.*?)-')

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
                weight = 0.95
                for step in range(1, len(raw_np)):
                    smooth_np[step] = (1 - weight) * raw_np[step] + weight * smooth_np[step - 1]
                # Add np_array to df
                df.loc[(df['description'] == i) & (df['seed'] == j) & (df['tag'] == k), 'smoothed'] = smooth_np.tolist()

    print(df.keys())
    print(df["smoothed"])

    # Plot the smoothed results using multi processing
    procs = 32  # Amount of processes/cores you want to use
    mp.set_start_method('spawn')  # This will make sure the different workers have different random seeds
    P = mp.cpu_count()  # Amount of available procs
    procs = max(min(procs, P), 1)  # Clip amount of procs to [1;P]

    def param_gen():
        # This function should return (yield because it's a generator) all parameter combinations you want to try
        for parameter in df['parameter_tested'].unique():
            for tag_value in df['tag'].unique():
                yield parameter, tag_value, df

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

