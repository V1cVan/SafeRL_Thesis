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
# tensorboard dev upload --logdir "./logfiles/Baselines/train" --name "Baselines" --description "Training of DQN_ER, DQN_PER, DDQN_ER, DDQN_PER, D3QN_ER, D3QN_PER with optimal parameters of the DDQN sweep."
# tensorboard dev upload --logdir "./logfiles/Deepset_tuning/train" --name "Deepset tuning" --description "Tuning of the deepset network. Defaults: DDQN ER defaults; Phi size=(32,32), Rho size=(32,32), Phi activation=relu, Rho activation=relu, Batch norm=False."
# tensorboard dev upload --logdir "./logfiles/Deepset_tuning/train" --name "Deepset tuning sweep 2" --description "Tuning of the deepset network sweep number 2. Defaults: DDQN ER defaults; Phi size=(64,64), Rho size=(32,32,32), Phi activation=Relu, Rho activation=Elu, Batch norm=False."
# tensorboard dev upload --logdir "./logfiles/DDQN_ER_reward_tuning/train" --name "DDQN reward shaping" --description "Tuning of the reward function to see if the following distance penalty is causing problems. Short run to 800 episodes. Default param"

# tensorboard dev upload --logdir "./logfiles/Deepset_tuning_fixed/train" --name "Fixed Deepset parameter sweep" --description "Tuning of deepset after model was fixed. Defaults: Phi/Rho size = (32,32), ActFunc = Relu, Batchnorm=False"
# tensorboard dev upload --logdir "./logfiles/Deepset_tuning_fixed/train" --name "Fixed Deepset parameter sweep number 2!" --description "Tuning of deepset run 2 after model was fixed. Defaults: Phi size=(32,32,32), Phi_act=relu, Rho_size = (32,32), Rho_act=tanh, Batchnorm=False"
# tensorboard dev upload --logdir "./logfiles/Deepset_tuning_original/train" --name "Old Deepset model (broken)" --description "Old default deepset model but re-run with after the reward function was tuned. For comparison to new deepset model."
# tensorboard dev upload --logdir "./logfiles/Deepset_baseline/train" --name "Final Deepset Baseline" --description "Final Fixed Deepset for baseline over 5 seeds. Default param: Phi_size=(32,64), Phi_act=Relu, Rho_size=(64,32), Rho_act=Tanh, BatchNorm=True."

# CNNs
# tensorboard dev upload --logdir "./logfiles/CNN1D_tuning_with_pooling/train" --name "1D CNN with pooling tuning sweep" --description "1D CNN sweep with pooling. Defaults: Pooling=True(sumpool). Only number of filters was tested."
# tensorboard dev upload --logdir "./logfiles/CNN1D_tuning_without_pooling/train" --name "1D CNN with OUT pooling tuning sweep" --description "1D CNN sweep with OUT pooling. Defaults: Pooling=True(sumpool). Only number of filters was tested."

# Generalisability baselines
# tensorboard dev upload --logdir "./logfiles/Generalisability_baselines/train" --name "Generalisability baselines - DDQN|Deepset|CNNs" --description "DDQN vs Deepset vs CNN w pooling vs CNN wo pooling. "
# tensorboard dev upload --logdir "./logfiles/Generalisability_baselines/test" --name "Generalisability baselines TEST - DDQN|Deepset|CNNs" --description "TEST - DDQN vs Deepset vs CNN w pooling vs CNN wo pooling. "
# tensorboard dev upload --logdir "./logfiles/Manually_create_generalisability_baselines/train" --name "Manually created generalisability baselines - not sure if it will work"
# tensorboard dev upload --logdir "./logfiles/Generalisability_baselines_rerun_with_fix/train" --name "Retrain of all the generalisability baselines. CNN with pooling now same units as without. And DDQN cannot see infinitely far. Padding on networks is valid. "
# tensorboard dev upload --logdir "./logfiles/Generalisability_baselines_rerun_with_fix/test" --name "TEST of all the generalisability baselines. CNN with pooling now same units as without. And DDQN cannot see infinitely far. Padding on networks is valid. "

# tensorboard dev upload --logdir "./logfiles/Generalisability_double_run_few_veh/train" --name "Generalisability train FEW vehicles" --description "Rerun of the generalisability tests but with a training scenario of 30 vehicles."
# tensorboard dev upload --logdir "./logfiles/Generalisability_double_run_many_veh/train" --name "Generalisability train MANY vehicles" --description "Rerun of the generalisability tests but with a training scenario of 70 vehicles."


# tensorboard dev upload --logdir "./logfiles/Generalisability_double_run_few_veh/test" --name "Generalisability TEST FEW vehicles" --description "Rerun of the generalisability tests but with a training scenario of 30 vehicles."
# tensorboard dev upload --logdir "./logfiles/Generalisability_double_run_many_veh/test" --name "Generalisability TEST MANY vehicles" --description "Rerun of the generalisability tests but with a training scenario of 70 vehicles."

# Temporal models tuning
# tensorboard dev upload --logdir "./logfiles/Temporal_tuning/train" --name "Temporal Tuning LSTM & 2D CNN" --description "LSTM & 2D CNN tuning sweeps. "

def start_run(parameter, tag_value, df, fig_path, run_type):

    sns.set_style("darkgrid")
    # if run_type != "parameter_sweep":
    sns.set(rc={'figure.figsize': (12, 5)})
    print(f"Plotting parameter={parameter}, tag value={tag_value}...")
    # Start your training loop using the given arguments
    plot_df = df.loc[(df['parameter_tested'] == parameter) & (df['tag'] == tag_value)]
    # plot_df = plot_df.iloc[::10,:] #Clearer by not plotting every step
    if run_type == 'generalisability':
        ax = sns.lineplot(x="step", y="smoothed",
                      hue="description",
                      data=plot_df,
                      alpha=0.8,
                      style="description",
                      markers=True,
                      dashes=False,
                      linewidth=2.2)
    else:
        ax = sns.lineplot(x="step", y="smoothed",
                          hue="description",
                          data=plot_df,
                          alpha=0.6,
                          linewidth=1.8)
    if tag_value == "Epsilon":
        ax.get_legend().remove()
    elif tag_value == "Beta increment":
        ax.get_legend().remove()
    else:
        if run_type == "parameter_sweep":
            ax.legend(loc='lower right', title='Parameter:')
        elif run_type == 'generalisability':
            ax.legend(loc='lower left', title=None)
        else:
            ax.legend(loc='lower right', title=None)

    if tag_value == "Vehicle speed":
        ax.set(ylim=(78, 108))
        ax.set(ylabel=tag_value + ' (km/h)')
        ax.set(xlabel="Episode")
    elif tag_value == "Reward":
        ax.set(ylim=(0.69, 0.95))
        ax.set(ylabel=tag_value)
        ax.set(xlabel="Episode")
    elif tag_value == "Average episode speed":
        # ax.set(xlim=(0, 95))
        ax.set(ylabel="Average speed")
        ax.set(xlabel="Number of vehicles")
    elif tag_value == "Average episode reward":
        # ax.set(xlim=(0, 95))
        ax.set(ylabel="Average reward")
        ax.set(xlabel="Number of vehicles")
    else :
        ax.set(ylabel=tag_value)
        ax.set(xlabel="Episode")

    plt_name = fig_path + "/" + parameter + '-' + tag_value
    plt.savefig(plt_name+'.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plt_name+'.png', dpi=300, bbox_inches='tight')
    print(f" Done with: Param/Method={parameter} ; Tag value={tag_value}")
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
def extract_columns(run_type, df, column1='seed', column2='description'):
    # Extract the seed to a column
    df[column1] = df['run'].str.extract(r'(Seed\d\d\d)')  # Seed must be three digits
    df[column1] = df[column1].str[4:]
    df[column1]  = df[column1]
    df[column1] = df[column1].astype('category')
    # Can access the different categories using shortened_df['seed'].dtypes.categories[index]

    # Extract the run description to a column
    if run_type == "parameter_sweep":
        # For parameter sweep comparison
        df[column2] = df['run'].str.extract(r'(Details=.*)')
        df[column2] = df[column2].str[8:]
    elif run_type == 'generalisability':
        # df[column2] = df['run'].str.extract(r'(Details=.*)')
        # df[column2] = df[column2].str[10:]
        df[column2] = df['run'].str.extract(r'( =.*)')
    else:
        # For method comparison:
        df[column2] =df['run'].str.extract(r'( =.*)')

    return df

def preprocess_data(csv_path, run_type):
    # (Re) Load the dataset
    df = pd.read_csv(csv_path)
    assert df.shape is not None

    # Remove variables that are not Episode reward or Vehicle velocity
    df = extract_columns(run_type, df)

    if run_type == 'parameter_sweep':
        # For parameter comparison:
        df['parameter_tested'] = df['description'].str.extract(r'(.*?)=')
        df = df.loc[df['tag'].isin(['Average episode reward', 'Average episode speed'])]
    elif run_type == 'generalisability':
        df['parameter_tested'] = df['description'].str.extract(r'(.*?)=')
        df['description'] = df['description'].str.extract(r'=(.*)')
        df = df.loc[df['tag'].isin(['Average episode speed', 'Average episode reward'])]
    else:
        # For method comparison:
        df['parameter_tested'] = df['description'].str.extract(r'(.*)=')
        df['description'] = df['description'].str.extract(r'=(.*)')
        df = df.loc[df['tag'].isin(['Average episode reward', 'Average episode speed'])]


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
                if (k != "Epsilon"):
                    if ((k == 'Average episode speed') or k == ('Average episode reward')):
                        weight = 0
                    else:
                        weight = 0.93
                else:
                    weight = 0.0
                for step in range(1, len(raw_np)):
                    smooth_np[step] = (1 - weight) * raw_np[step] + weight * smooth_np[step - 1]
                # Add np_array to df
                df.loc[(df['description'] == i) & (df['seed'] == j) & (df['tag'] == k), 'smoothed'] = smooth_np.tolist()

    print(f"Values smoothed by factor {weight}...")

    if run_type != 'generalisability':
        # Extract the peak mean, overall mean, and overall std. over the seeds for each parameter and metric tested:
        metrics_columns = ['metric', 'parameter', 'description', 'average_mean', 'peak_mean', 'final_mean', 'average_std']
        metrics_data = []
        for i in df['description'].unique():  # Loop through parameters
                for j in df['tag'].unique():  # Loop through Mean episode reward or mean vehicle speed
                    if j != 'Epsilon':
                        df_selection = df.loc[(df['description'] == i) & (df['tag'] == j)]

                        mean_line = df_selection.groupby('step')['smoothed'].mean()
                        average_mean = mean_line.mean()
                        final_mean = mean_line[mean_line.index[-1]]
                        peak_mean = mean_line.max()
                        std_line = df_selection.groupby('step')['smoothed'].std()
                        average_std = std_line.mean()

                        parameter = df_selection['parameter_tested'].unique()[0]
                        metrics_data.append([j, parameter, i, average_mean, peak_mean, final_mean, average_std])
        metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
        excel_path = csv_path[:-4] + "_metrics.xlsx"
        metrics_df.to_excel(excel_path)

    print(df.keys())
    print(df["smoothed"])

    return df

def display_results_summary(csv_path):
    metrics_df = pd.read_excel(csv_path[:-4] + "_metrics.xlsx")

    reward_df = metrics_df.loc[metrics_df['metric'] == 'Average episode reward']
    velocity_df = metrics_df.loc[metrics_df['metric'] == 'Average vehicle speed']

    vel_data = {'average_mean':[], 'peak_mean':[], 'final_mean':[], 'average_std':[]}
    rew_data = {'average_mean':[], 'peak_mean':[], 'final_mean':[], 'average_std':[]}
    for parameter in metrics_df['parameter'].unique():
        for item in list(vel_data.keys()):
            rew_selection = reward_df.loc[(reward_df['parameter'] == parameter)]
            vel_selection = velocity_df.loc[(velocity_df['parameter'] == parameter)]
            if item != 'average_std':
                rew_result = rew_selection.loc[(rew_selection[item] == rew_selection[item].max())]
                vel_result = vel_selection.loc[(vel_selection[item] == vel_selection[item].max())]
            else:
                rew_result = rew_selection.loc[(rew_selection[item] == rew_selection[item].min())]
                vel_result = vel_selection.loc[(vel_selection[item] == vel_selection[item].min())]

            rew_data[item].append(str(rew_result["description"].values[0]) +' | '+ str(rew_result[item].values[0]))
            vel_data[item].append(str(vel_result["description"].values[0]) +' | '+ str(vel_result[item].values[0]))

    best_reward_df = pd.DataFrame(rew_data)
    rew_path = csv_path[:-4] + "_best_reward_params.xlsx"
    best_reward_df.to_excel(rew_path)
    best_velocity_df = pd.DataFrame(vel_data)
    vel_path = csv_path[:-4] + "_best_velocity_params.xlsx"
    best_velocity_df.to_excel(vel_path)

    print("Best parameters i.t.o. reward: \n")
    print(rew_data)
    print("Best parameters i.t.o. velocity: \n")
    print(vel_data)

    print("Saving best parameters to xlsx...")


if __name__ == "__main__":
    experiment_ids = {
        "DDQN_ER_initialisers": "8viGslanQLWtMaORGizRaQ",
        "DDQN_ER_tuning": "ndalAuYXRxGuML7vkgPA4Q",
        "Epsilon_sweep": "t1GY8hx8QFWiNjyofqJUNw",
        "DQN_DDQN_standardisation_short": "0o0v4ZgrTWWUnOUlBrWmow",
        "DQN_DDQN_standardisation_long": "rhGbDukaTL28a8hjZO6QiQ",
        'Baselines': 'xCB7wzRWTRKhrrv10HkKcw',
        'Deepset_tuning_run1': 'foWXaOj4Rvy1i78j3HnepA',
        'Deepset_tuning_run2': 'daXDaA0rRYWXofnUEM5R3Q',
        'DDQN_reward_shaping': 'S0cm843GQC66rgyfPKfnQA',
        'CNN_normal_tuning': 'x',
        'Deepset_old': 'oc7AyPI8Q9eKRrUvNYsBtA',
        'Deepset_tuning_fixed_run1': '6HopvVnTR2GEzYUkBdTdDQ',
        'Deepset_tuning_fixed_victor': 'yRJrRW96SVCE8OwN0zV1qw',
        'Deepset_baseline': '2Y7mEYXnQOGtqEPKDLd7aA',
        'CNN_tuning_without_pooling': 'denVBYeVSzmfNduXpOnC3A',
        'CNN_tuning_with_pooling': 'EiG3S0LRRNaUkotA8tmkxQ',
        'Generalisability_baselines': 'k11SRrn3TMmXg7Zrn1Axpw',
        'Generalisability_test': 'rjY2gnrMTGmyMG0r9YCRiw',
        'Generalisability_train_few_vehicles': 'fB2Wr2SLQ1ilN8DQWyuZ7g',
        'Generalisability_train_many_vehicles': 'OmSOxVxZT4ipNOZK7yagHA',
        'Generalisability_test_few_vehicles': 'RWAWXXCqTBSbmBwOWIWnAw',
        'Generalisability_test_many_vehicles': 'V88XQ5ZvTdGjhv9MNz4aJA',
        'Temporal_tuning': 'BaCW51LPQ9WTNgabD1Jblw',
        'DDQN_IBM_RULE_BASED': "i9NxLq4nSzaNBSb7WHbYLw",

        'Noise_tuning': "Lt8ug0w3TraMLGl9CQmhgA",
        'Varied_random_training_scenario': "Wao8WzNlTzKJuWreqx2PPA",
        'Velocity_removal_baseline_default': "FXaEoDl5TGmNMogAWG2gEw",
        'Velocity_removal_baseline_extended': 'WhNIRNFPQ2mq1e2Lx1WiWw',
        'Velocity_removal_baseline_many_fast': '3QTRa4mSQ5Gvzb8DUr27sQ',
        'Velocity_removal_baseline_stress': 'jn1yZJFMQxC4MgNnSa4QQQ',
        'Temporal_without_state_velocity': "bSPId35rRGSni5rWz5vH2Q",
        'Temporal_with_state_velocity': 'sjX3MBf1QRGAOZUJ92ef6g',
        'Noise': '1XKKNjIERs6G5tvSPbrcng'
    }

    experiment_paths = {
        "DDQN_ER_initialisers": "./logfiles/DDQN_ER_initialisers",
        "DDQN_ER_tuning": "./logfiles/DDQN_ER_tuning_run2/train",
        "Epsilon_sweep": "./logfiles/DDQN_Epsilon_Beta_tuning/train",
        "DQN_DDQN_standardisation_short": "./logfiles/DQN_DDQN_standardisation_target_update/train",
        "DQN_DDQN_standardisation_long": "./logfiles/DQN_DDQN_standardisation_target_update_long_run/train",
        'Baselines': './logfiles/Baselines/train',
        'Deepset_tuning_run1': './logfiles/Deepset_tuning_run1/train',
        'Deepset_tuning_run2': './logfiles/Deepset_tuning_run2/train',
        'DDQN_reward_shaping': './logfiles/DDQN_ER_reward_tuning/train',
        'CNN_normal_tuning': './logfiles/CNN_normal_tuning/train',
        'Deepset_old': './logfiles/Deepset_tuning_original/train',
        'Deepset_tuning_fixed_run1': './logfiles/Deepset_tuning_fixed_run1/train',
        'Deepset_tuning_fixed_victor': './logfiles/Deepset_tuning/train',
        'Deepset_baseline': './logfiles/Deepset_baseline/train',
        'CNN_tuning_without_pooling': './logfiles/CNN1D_tuning_without_pooling/train',
        'CNN_tuning_with_pooling': './logfiles/CNN1D_tuning_with_pooling/train',
        'Generalisability_baselines': './logfiles/Generalisability_baselines/train',
        'Generalisability_test': './logfiles/Generalisability_baselines/test',
        'Generalisability_train_few_vehicles' : './logfiles/Generalisability_double_run_few_veh/train',
        'Generalisability_train_many_vehicles': './logfiles/Generalisability_double_run_many_veh/train',
        'Generalisability_test_few_vehicles': './logfiles/Generalisability_double_run_few_veh/test',
        'Generalisability_test_many_vehicles': './logfiles/Generalisability_double_run_many_veh/test',

        'Temporal_tuning': './logfiles/Temporal_tuning/train',
        'DDQN_IBM_RULE_BASED': "./logfiles/Baselines_IBM_RULE_BASED",
        'Noise_tuning': "./logfiles/Noise_tuning/train",
        'Varied_random_training_scenario': "./logfiles/Varied_random_training_scenario/train",
        'Velocity_removal_baseline_default': "./logfiles/Velocity_removal_baseline_default/train",
        'Velocity_removal_baseline_extended': './logfiles/Velocity_removal_baseline_extended/train',
        'Velocity_removal_baseline_many_fast': './logfiles/Velocity_removal_baseline_many_fast/train',
        'Velocity_removal_baseline_stress': './logfiles/Velocity_removal_baseline_stress/train',
        'Temporal_without_state_velocity': "./logfiles/Temporal_networks_for_report/without_state_velocity",
        'Temporal_with_state_velocity': './logfiles/Temporal_networks_for_report/with_state_velocity',
        'Noise': './logfiles/Noise_tuning/train'
    }

    experiment_names = list(experiment_ids.keys())
    print(experiment_names)

    experiment_name = "Generalisability_test_few_vehicles"
    run_type = 'method_comparison'  # 'method_comparison' OR 'parameter_sweep' or 'generalisability'
    csv_path = experiment_paths[experiment_name] + "/" + experiment_name + '.csv'
    download_and_save(experiment_name, csv_path)  # Comment out if you don't want to re-download the data

    # Pre-process (smooth data from csv)
    df = preprocess_data(csv_path, run_type)

    # Display the best results of the different parameter:
    if run_type != 'generalisability':
        display_results_summary(csv_path)

    # Plot the smoothed results using multi processing
    procs = 1   # Amount of processes/cores you want to use
    mp.set_start_method('spawn')  # This will make sure the different workers have different random seeds
    P = mp.cpu_count()  # Amount of available procs
    procs = max(min(procs, P), 1)  # Clip amount of procs to [1;P]

    def param_gen():
        # This function should return (yield because it's a generator) all parameter combinations you want to try
        for parameter in df['parameter_tested'].unique():
            for tag_value in df['tag'].unique():
                yield parameter, tag_value, df, experiment_paths[experiment_name], run_type

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
