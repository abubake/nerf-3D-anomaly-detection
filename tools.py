import torch
import os
import configparser
import numpy as np
import pyvista as pv
import zipfile
import json
import random
import shutil
from typing import List
import ast

from datagen import generate_data

def setup_experiment_folders(experiment_name):
    '''
    Creates experiment directories in the experiments folder:

    experiments
        - experiment_name
            - figures
            - models

    '''
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


def load_experiment_models(config_path: str, device: str ='cuda') -> dict[List[str]]:
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
        single_test = int(config['TESTS']['single_test'])
    except KeyError as e:
        print(f"Missing configuration key: {e}")

    return load_models_from_experiments(single_test, base_directory, experiment_name, device='cuda')


def load_models_from_experiments(single_test: int = 0, base_directory: str = "experiments", experiment_name: str = "test", device: str = "cuda") -> dict[List[str]]:
        '''
        Loads training models from experiment folder.
        '''
        if single_test == 1:
            experiment_directory = os.path.join(os.path.join(base_directory, experiment_name), 'models')
            print(f"pulling models from directory: {experiment_directory}")
            exit("single test not implmented")
        else:
            experiment_directory: List[str] = []
            model_folders: List[str] = []

            for folder in os.listdir(os.path.join('data', experiment_name)):
                model_folders.append(folder)
                print(f"current folder: {folder}")
                experiment_directory.append(os.path.join(os.path.join(base_directory, experiment_name),
                                                          os.path.join(folder, 'models')))

        # Initialize a dict to store the loaded models or data
        loaded_models: dict = {key: [] for key in model_folders}

        # Iterate over all models in experiment directory (for ensembles)
        for i, exp in enumerate(model_folders):
            for filename in os.listdir(experiment_directory[i]):
                if filename.endswith('.pth'):
                    file_path = os.path.join(experiment_directory[i], filename)
                    try:
                        model_i = torch.load(file_path).to(device)
                        # Append the loaded model or data to the list
                        loaded_models[exp].append(model_i)
                        print(f"Loaded {filename} successfully.")
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")

        return loaded_models

def uncertainty_plot(scalar_field=None,scalars=None, pts=None,
                                only_mask=True, threshold=0.1,
                                  plot=True, plotTitle='lookatthisgraph'):
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
        if plot == True:
            plotter = pv.Plotter(title=plotTitle)
            plotter.add_points(np.column_stack((x_coords, y_coords, z_coords)), scalars=values, cmap="inferno")
            plotter.show()

        return np.stack((x_coords, y_coords, z_coords), axis=-1), values
    
    else:
        output_array = np.zeros((100, 100, 100))

        for i, (x, y, z) in enumerate(pts):
            output_array[x, y, z] = scalars[i]

        # # Find the coordinates of points above the threshold
        above_threshold_mask = output_array > threshold
        x_coords, y_coords, z_coords = np.where(above_threshold_mask)

        # Extract the values of points above the threshold
        values = output_array[above_threshold_mask]

        if plot == True:
            plotter = pv.Plotter(title=plotTitle)
            plotter.add_points(np.column_stack((x_coords, y_coords, z_coords)), scalars=values, cmap="inferno")
            plotter.show()

        return np.column_stack((x_coords, y_coords, z_coords))
    
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
    ################ READING IN CONFIG PARAMS #############################

    anomaly_image_per_set_str = tests_params.get('anomaly_image_per_set', '[]')
    try:
        anomaly_image_per_set = ast.literal_eval(anomaly_image_per_set_str)

        if not isinstance(anomaly_image_per_set, list):
            raise ValueError("anomaly_image_per_set is not a list")
        
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing anomaly_image_per_set: {e}")
        anomaly_image_per_set = []

    generate_single_dataset = int(tests_params['single_test'])
    single_test_anomaly_imgs = int(tests_params['single_test_anomaly_imgs']) # TODO: make this entered in terminal at runtime

    ################### BEGIN DATA CREATION #############################

    zip_path1 = "data_zips/"+experiment_name+".zip"
    zip_path2 = "data_zips/"+experiment_name+"_missing.zip"

    # Step 1: Check if there is already experiment data created.
    if os.path.isdir("data/"+experiment_name):
        print(f"- Training data for running the experiment already exists, or another experiment already has the name '{experiment_name}'")
    else:
        # Create new folder "experiment_name"
        os.makedirs("data/"+experiment_name)
        #assert generate_single_dataset == 1
        if generate_single_dataset == 1:

            set_i = f"data/{experiment_name}/set{single_test_anomaly_imgs}"
            os.makedirs(set_i, exist_ok=True)
            create_dataset_i(zipFilepathSet1=zip_path1, zipFilepathSet2=zip_path2, currentDataset=set_i,
                                originalImgNum=200, anomalyImgNum=single_test_anomaly_imgs)
        else:
            # generate multiple datasets w/ different amu of anomaly images
            for i in anomaly_image_per_set:
                set_i = f"data/{experiment_name}/set{i}"
                #print(f"Anomaly images in {set_i}:{i}")

                os.makedirs(set_i, exist_ok=True)
                create_dataset_i(zipFilepathSet1=zip_path1, zipFilepathSet2=zip_path2, currentDataset=set_i,
                                originalImgNum=200, anomalyImgNum=i)

        print("- Required datasets created for the experiment")


def create_dataset_i(zipFilepathSet1="", zipFilepathSet2="", currentDataset="", originalImgNum=200, anomalyImgNum=10):
    '''
    Creates a dataset for given parameters from two exisiting zip files
    '''
    m = anomalyImgNum
    n = originalImgNum - anomalyImgNum

    # select data randomly from old json to put into the new
    data = read_json_from_zipfile(zipFilepath=zipFilepathSet1, jsonFilename="transforms_train.json")
    new_frames = random.sample(data["frames"], n) # samples w/o replacement
    image_names = read_images_from_zipfile(zipFilepathSet1)
    filtered_image_names = [name for name in image_names if name in {os.path.basename(entry['file_path']) for entry in new_frames}]
    move_images_from_zip(filesToMove=filtered_image_names, source=zipFilepathSet1, destination=currentDataset+"/train")
    assert len(new_frames) == n
    print(f"len of new frames:{len(new_frames)}")

    updated_frames = rename_images_and_corresponding_dictionary_poses(directory=currentDataset+"/train", data_dict=new_frames, start_index=1)
    assert len(new_frames) == len(updated_frames)
    print(f"len of updated frames:{len(new_frames)}")

    # repeat and append for anomaly
    data_anom = read_json_from_zipfile(zipFilepath=zipFilepathSet2, jsonFilename="transforms_train.json")
    new_frames_anom = random.sample(data_anom["frames"], m)
    image_names_anom = read_images_from_zipfile(zipFilepathSet2)
    print(f"anom image count for {currentDataset}: {len(image_names_anom)}")

    filtered_image_names_anom = [name for name in image_names_anom if name in {os.path.basename(entry['file_path']) for entry in new_frames_anom}]
    print(f"combined pose frame length: {len(new_frames)+len(new_frames_anom)}")
    print(f"combined image length: {len(filtered_image_names)+len(filtered_image_names_anom)}")

    move_images_from_zip(filesToMove=filtered_image_names_anom, source=zipFilepathSet2, destination=currentDataset+"/train_anom")

    anom_frames = rename_images_and_corresponding_dictionary_poses(directory=currentDataset+"/train_anom", data_dict=new_frames_anom, start_index=n+1)
    assert len(anom_frames) == m

    updated_frames.extend(anom_frames)
    wrapped_frames = {"frames": updated_frames}
    write_dict_to_json(dictionary=wrapped_frames, file_path=currentDataset+"/transforms_train.json")

    # Last Step: Cleanup
    source_folder = currentDataset+"/train_anom"
    destination_folder = currentDataset+"/train"
    [shutil.move(os.path.join(source_folder, f), destination_folder) for f in os.listdir(source_folder)]

    source_folder = currentDataset+"/train"
    destination_folder = currentDataset
    [shutil.move(os.path.join(source_folder, f), destination_folder) for f in os.listdir(source_folder)]
    
    os.rmdir(currentDataset+"/train_anom")
    os.rmdir(currentDataset+"/train")

    generate_data(using_blender=True, training_split=1.0, img_folder="imgs", focal=120, width=400, project_data_dir=currentDataset)
    

def rename_images_and_corresponding_dictionary_poses(directory: str, data_dict: dict, start_index: int = 1) -> dict:
    '''
    Renames images and the poses which correspond in a dictionary
    '''
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    image_files.sort()
    #print(f"sorted image files:{image_files} length:{len(image_files)}")

    # create a temporary directory for the transfer
    temp_dir = f"{directory}/temp"
    os.mkdir(path=temp_dir)

    for entry in data_dict:
                entry["updated"] = False
    
    # Loop through each image file and rename it
    count: int = 0
    try:
        for i, old_name in enumerate(image_files, start=start_index):

            new_name = f"{i:04}.png"
            old_path = os.path.join(directory, old_name)
            temp_path = os.path.join(temp_dir, new_name)

            for entry in data_dict:
                if not entry["updated"]:
                    old_file_name = os.path.basename(entry["file_path"])
                    if old_file_name == old_name:
                        #print(f"oldfilename:{old_file_name} to newfilename: {new_name}")
                        entry["file_path"] = new_name
                        entry["updated"] = True
                        break

            try:
                shutil.move(old_path, temp_path)

            except Exception as e:
                    print(f"Error moving {old_path} to {temp_path}: {e}")
                    continue
            
        # Move files back to original directory with new name
        for new_name in os.listdir(temp_dir):
            temp_path = os.path.join(temp_dir, new_name)
            final_path = os.path.join(directory, new_name)
                
            # Check if the final path already exists and handle it
            if os.path.exists(final_path):
                print(f"Warning: {final_path} already exists. Skipping.")
                continue
            
            try:
                shutil.move(temp_path, final_path)
                count += 1
            except Exception as e:
                print(f"Error moving {temp_path} to {final_path}: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Move executed {count} times, and moved {len([f for f in os.listdir(directory) if f.endswith('.png')])} images!")
        return sorted(data_dict, key=lambda x: x["file_path"])


def write_dict_to_json(dictionary, file_path):
    """
    Writes a dictionary to a new JSON file.

    :param dictionary: The dictionary to write to the JSON file.
    :param file_path: The path to the JSON file to create.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)  # indent=4 for pretty formatting
        print(f"Dictionary successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def move_images_from_zip(filesToMove=[], source="", destination=""):
    '''
    Moves images from a source zip folder to a destination folder
    '''
    os.makedirs(destination, exist_ok=True)

    with zipfile.ZipFile(source, 'r') as zip_ref:
        train_files = [f for f in zip_ref.namelist() if f.startswith('train/')]
        files_to_move = [f for f in train_files if os.path.basename(f) in filesToMove]

        for file in files_to_move:
            temp_path = zip_ref.extract(file, path='/tmp')
            destination_path = os.path.join(destination, os.path.basename(file))
            shutil.move(temp_path, destination_path)


def read_images_from_zipfile(zipFilepath=""):
    '''
    Reads images from a zipfile to get a list of names
    '''
    with zipfile.ZipFile(zipFilepath, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        png_filenames = [
            os.path.basename(file)  # Get the base filename (excluding the directory path)
            for file in all_files
            if file.startswith('train/') and file.lower().endswith('.png')
        ]
    return png_filenames


def read_json_from_zipfile(zipFilepath="", jsonFilename=""):
    '''
    Reads the data within a given json file within a zipfile and drops it
    in a variable

    OUTPUTS:
        - data (Type: dictionary)
    '''
    with zipfile.ZipFile(zipFilepath, 'r') as zip_ref:
            if jsonFilename in zip_ref.namelist():
                with zip_ref.open(jsonFilename) as json_file:
                    json_bytes = json_file.read()
                    json_str = json_bytes.decode('utf-8')
                    data = json.loads(json_str)
                    return data
            else:
                print(f"{jsonFilename} not found in the ZIP archive.")
                return None
            

if __name__ == '__main__':
    exp_dict = load_experiment_models(config_path="configs/test.conf", device='cuda')
    print(exp_dict)
    print("test success!")