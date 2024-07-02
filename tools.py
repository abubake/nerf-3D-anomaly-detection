import torch
import os
import configparser
import numpy as np
import pyvista as pv

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


def load_experiment_models(config_path, device='cuda'):
    """
    Load all .pth files from the specified experiment directory using configuration details from a .conf file.

    :param config_path: Path to the configuration .conf file.
    :return: A list of loaded models or data from the .pth files.
    """
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration from the .conf file
    config.read(config_path)

    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} does not exist.")
    elif os.path.getsize(config_path) == 0:
        print(f"Configuration file {config_path} is empty.")
    else:
        print(f"Configuration file {config_path} is found and has content.")

    # Print out all sections and keys for debugging
    print("Sections found in config:", config.sections())
    for section in config.sections():
        print(f"Keys in section '{section}':", config.options(section))

    # Extract the base directory and experiment name from the config
    try:
        base_directory = config['EXPERIMENT']['base_directory']
        experiment_name = config['EXPERIMENT']['experiment_name']
    except KeyError as e:
        print(f"Missing configuration key: {e}")
        return []

    # Construct the full path to the experiment directory
    experiment_directory = os.path.join(os.path.join(base_directory, experiment_name), 'models')

    # Initialize a list to store the loaded models or data
    loaded_models = []

    # Check if the directory exists
    if not os.path.isdir(experiment_directory):
        print(f"Directory {experiment_directory} does not exist.")
        return loaded_models

    # Iterate over all files in the experiment directory
    for filename in os.listdir(experiment_directory):
        # Check if the file has a .pth extension
        if filename.endswith('.pth'):
            # Construct the full path to the .pth file
            file_path = os.path.join(experiment_directory, filename)

            # Load the .pth file
            try:
                loaded_model = torch.load(file_path).to(device)

                # Append the loaded model or data to the list
                loaded_models.append(loaded_model)
                print(f"Loaded {filename} successfully.")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    return loaded_models

def uncertainty_plot(scalar_field=None,scalars=None, pts=None,
                                only_mask=True, threshold=0.1):
    '''
    Uses pyvista to plot values of 3D points. Two options available for plotting.
    Can provide scalars and points, or scalar field (3D points with scalar values
    defined at each point).

    If only_mask = True, then only scalars that match the given points are used.
    '''
    # Extract the values of points above the threshold
    if scalar_field is not None:

        above_threshold_mask = scalar_field > threshold
        x_coords, y_coords, z_coords = np.where(above_threshold_mask)

        # Extract the values of points above the threshold
        values = scalar_field[above_threshold_mask]

        # Create a PyVista Plotter
        plotter = pv.Plotter(title="my plot")

         # Add the points to the plotter
        plotter.add_points(np.column_stack((x_coords, y_coords, z_coords)), scalars=values, cmap="inferno")
        plotter.show()
        return np.stack((x_coords, y_coords, z_coords), axis=-1), values
    
    else:
        output_array = np.zeros((100, 100, 100))

        for i, (x, y, z) in enumerate(pts):
            output_array[x, y, z] = scalars[i]

        # Create a PyVista Plotter
        plotter = pv.Plotter(title="my plot")

        # # Find the coordinates of points above the threshold
        above_threshold_mask = output_array > threshold
        x_coords, y_coords, z_coords = np.where(above_threshold_mask)

        # Extract the values of points above the threshold
        values = output_array[above_threshold_mask]

        # Add the points to the plotter
        plotter.add_points(np.column_stack((x_coords, y_coords, z_coords)), scalars=values, cmap="inferno")
        plotter.show()
        return 0
    
def create_experiment_data(experiment_name='test_experiment', tests_params={}):
    '''
    Given experiment name, takes related data_zips and
    creates the datasets needed for the experiment,
    as specified in the corresponding config file,
    given they don't already exist.

    INPUT: 
        - experiment_name: name of the experiment (string)
        - kd_tests_params: dictonary of variables in TEST section of the experiment's
        config file (dictionary)


    A prerequiste to use of this function is generation of the original and
    changed scene images and corresponding poses in blender,
    and placement into the "data_zips" folder.

    The function will search for a keyword for the data, EX: pig.zip
    Then it will search for the scene 2 zip, expecting a name of
    EX: pig_missing.zip.

    The experiment name you select must match the dataset name you generated in blender
    (EX: pig)
    '''
    # Step 1: Check if there is already experiment data created.
    if os.path.isdir("data/"+experiment_name):
        print(f"Experiment data already exists, or another experiment already has the name '{experiment_name}'")
    else:
        os.makedirs("data/"+experiment_name)

        # FIRST CASE: experiment with seperate datasets for different ratios of original to anomaly images
        max_anomaly_images = tests_params['max_anomaly_images']
        step_size = tests_params['step_size']

        # GENERAL PROCESS:
        # Create new folder "experiment_name"
        # Pull 

        print("required datasets created for the experiment")