import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorboard as tb
import numpy as np

"""
For uploading experiment to Tensorboard:
    tensorboard dev upload 
        --logdir=="./logfiles/TRAIN OR TEST/experiment 
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


if __name__ == "__main__":
    experiment_ids = {
        "DDQN_ER_initialisers": "8viGslanQLWtMaORGizRaQ",
        "DDQN_ER_tuning": "aZQNhVtwTwqAamNCq1UKZg"
    }
    experiment_paths = {
        "DDQN_ER_initialisers": "./logfiles/DDQN_ER_initialisers",
        "DDQN_ER_tuning": "./logfiles/DDQN_ER_tuning/train"
    }

    # Retrieve the training data from Tensorboard Dev
    experiment_names = list(experiment_ids.keys())

    experiment_id = experiment_ids["DDQN_ER_tuning"]
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    results = experiment.get_scalars()

    # Save downloaded data to a csv file
    csv_path = experiment_paths["DDQN_ER_tuning"] + "/" + experiment_names[1] + '.csv'
    results.to_csv(csv_path, index=False)
    results_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(results_df, results)
    print("Keys for dataframe: " + results_df.keys())

    # (Re) Load the dataset
    df = pd.read_csv(csv_path)
    assert df.shape is not None

    # Remove variables that are not Episode reward or Vehicle velocity
    df = extract_columns(df)
    df = df.loc[df['tag'].isin(['Mean episode reward', 'Mean vehicle speed for episode'])]
    df['parameter_tested'] = df['description'].str.extract(r'(.*?)-')

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

    # Plot the smoothed results
    sns.set_style("darkgrid")
    for parameter in df['parameter_tested'].unique():
        for tag_value in df['tag'].unique():
            plot_df = df.loc[(df['parameter_tested'] == parameter) & (df['tag'] == tag_value)]
            # plot_df = plot_df.iloc[::10,:] #Clearer by not plotting every step
            ax = sns.lineplot(x="step", y="smoothed",
                              hue="description",
                              data=plot_df,
                              alpha=0.6,
                              linewidth=0.2)
            ax.set_title(str(tag_value))
            plt.show()
            break  # Break after first parameter to check that this works.
