import os
import matplotlib.pyplot as plt

def setup_experiment_folders(experiment_name):
    # Define the path to the experiments directory relative to the current working directory
    experiments_dir = os.path.join(os.getcwd(), 'experiments')

    # Define the path to the specific experiment's folder within experiments
    experiment_dir = os.path.join(experiments_dir, experiment_name)

    # Define the path to the models directory within the specific experiment folder
    models_dir = os.path.join(experiment_dir, 'models')

    figures_dir = os.path.join(experiment_dir, 'figures')

    # Create the experiments directory if it doesn't exist
    if not os.path.exists(experiments_dir):
        print(f"Creating directory: {experiments_dir}")
        os.makedirs(experiments_dir)

    # Create the experiment_name directory within experiments if it doesn't exist
    if not os.path.exists(experiment_dir):
        print(f"Creating directory: {experiment_dir}")
        os.makedirs(experiment_dir)

    # Create the models directory within experiment_name if it doesn't exist
    if not os.path.exists(models_dir):
        print(f"Creating directory: {models_dir}")
        os.makedirs(models_dir)
    
    # Create the figures directory within experiment_name if it doesn't exist
    if not os.path.exists(figures_dir):
        print(f"Creating directory: {figures_dir}")
        os.makedirs(figures_dir)

    print("All required directories are set up.")

# def plot_and_save(training_loss, model_index, experiment_name, phase='training'):
#     '''
#     Plotting figures for training_nerf.py
#     '''
#     plt.plot(training_loss)
#     plt.title(f"Training Loss for Model {model_index + 1} ({phase.capitalize()})")
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
    
#     # Save the plot to a file
#     filename = f"experiments/{experiment_name}/plots/model_{model_index}_{phase}.png"
#     plt.savefig(filename)
#     print(f"Saved plot to {filename}")