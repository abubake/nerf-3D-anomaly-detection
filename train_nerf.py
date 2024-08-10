import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tools import setup_experiment_folders
from tools import create_experiment_data
from evaluate import 
from dataset import get_rays
from model import Nerf
from ml_helpers import training
import configparser
import argparse
import os

def main(args):

    config = configparser.ConfigParser()

    # Read the configuration file
    if not os.path.exists(args.conf):
        raise FileNotFoundError(f"The configuration file {args.conf} does not exist.")
    config.read(args.conf)

    # Access the parameters from the 'EXPERIMENT' section
    batch_size = config.getint('EXPERIMENT', 'batch_size')
    height = config.getint('EXPERIMENT', 'height')
    width = config.getint('EXPERIMENT', 'width')
    imgs = config.getint('EXPERIMENT', 'imgs')
    pth_file = config.get('EXPERIMENT', 'pth_file')
    experiment_name = config.get('EXPERIMENT', 'experiment_name')
    device = config.get('EXPERIMENT', 'device')
    tn = config.getfloat('EXPERIMENT', 'tn')
    tf = config.getfloat('EXPERIMENT', 'tf')
    nb_epochs = config.getint('EXPERIMENT', 'nb_epochs')
    lr = config.getfloat('EXPERIMENT', 'lr')
    gamma = config.getfloat('EXPERIMENT', 'gamma')
    nb_bins = config.getint('EXPERIMENT', 'nb_bins')
    ensembles = config.getint('EXPERIMENT', 'ensembles')

    # Acess all the parameters in the TESTS section
    section = 'TESTS'
    test_params = {}
    if section in config:
        test_params = {key: config.get(section, key) for key in config[section]}
    else:
        print(f"Section {section} not found in the {args.conf}")

    # Start here!
    create_experiment_data(experiment_name, test_params)
    setup_experiment_folders(experiment_name)
    train_models(experiment_name, ensembles,lr, gamma, tn, tf, nb_bins,
                  nb_epochs, batch_size, imgs, height, width, device)
    #post_processing()
    # evaluate_models()
    
    print("Experiment Complete!")

#########################################################################################
def evaluate_models():
    '''
    Currently evaluates all models in terms of 3d IoU, across multiple post-processing scenarios
    TODO: move this to the evaluate.py file
    '''
    ...

def post_processing():
    '''
    Sets up trained models for evaluations for the given task.
    '''


def train_models(experiment_name, ensembles,lr, gamma, tn, tf, nb_bins,
                  nb_epochs, batch_size, imgs, height, width, device):
    '''
    Trains all models specified for the experiment with given hyperparameters.

    args:
        x
    returns:
        
    '''

    directories = sorted([name for name in os.listdir(f"data/{experiment_name}") 
                          if os.path.isdir(os.path.join(f"data/{experiment_name}", name))])
    for directory in directories: # for each individual set of data in the dataset we are running, execute.
        
        print(f"- Now training dataset {directory}...")
        setup_experiment_folders(os.path.join(f"{experiment_name}", directory))
        
        directory_path = os.path.join(f"data/{experiment_name}", directory)
        dataloader, dataloader_warmup = load_rays(dataset_path=directory_path, mode="train", batch_size=batch_size, imgs=imgs, height=height, width=width)

        for i in range(ensembles): # execute same model desired number of times

            model = Nerf(hidden_dim=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

            training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, 1, dataloader_warmup, model_name=f"experiments/{experiment_name}/{directory}/models/M{i}.pth", device=device)
            phase = "warmup"
            plt.figure()
            plt.plot(training_loss)
            filename = f"experiments/{experiment_name}/{directory}/figures/model_{i}_{phase}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
            plt.close()
            
            training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, model_name=f"experiments/{experiment_name}/{directory}/models/M{i}.pth", device=device)
            phase = 'training'
            plt.figure()
            plt.plot(training_loss)
            filename = f"experiments/{experiment_name}/{directory}/figures/model_{i}_{phase}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
            plt.close()



def load_rays(dataset_path="", mode="train", batch_size=1024, imgs=200, height=400, width=400):
    
    o, d, target_px_values = get_rays(dataset_path, mode='train')
    print(f"- Image count: {imgs}")
    print(f"- Image Dimensions: Width:{width}, Height:{height}")

    dataloader = DataLoader(torch.cat((torch.from_numpy(o).reshape(-1, 3).type(torch.float),
                                    torch.from_numpy(d).reshape(-1, 3).type(torch.float),
                                    torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)), dim=1),
                        batch_size=batch_size, shuffle=True)

    dataloader_warmup = DataLoader(torch.cat((torch.from_numpy(o).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                                torch.from_numpy(d).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                                torch.from_numpy(target_px_values).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float)), dim=1),
                        batch_size=batch_size, shuffle=True)
    
    return dataloader, dataloader_warmup


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run NeRF training with specified configuration file.")
    parser.add_argument('--conf', type=str, default="./configs/conf.conf", help="Path to the configuration file")
    args = parser.parse_args()
    main(args)