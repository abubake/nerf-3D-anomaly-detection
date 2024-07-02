import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tools import setup_experiment_folders
from tools import create_experiment_data
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
    dataset_path = config.get('EXPERIMENT', 'dataset_path')
    pth_file = config.get('EXPERIMENT', 'pth_file')
    experiment_name = config.get('EXPERIMENT', 'experiment_name')
    device = config.get('EXPERIMENT', 'device')
    tn = config.getfloat('EXPERIMENT', 'tn')
    tf = config.getfloat('EXPERIMENT', 'tf')
    nb_epochs = config.getint('EXPERIMENT', 'nb_epochs')
    lr = config.getfloat('EXPERIMENT', 'lr')
    gamma = config.getfloat('EXPERIMENT', 'gamma')
    nb_bins = config.getint('EXPERIMENT', 'nb_bins')
    N = config.getint('EXPERIMENT', 'N')

    # Acess all the parameters in the TESTS section
    section = 'TESTS'
    test_params = {}
    if section in config:
        test_params = {key: config.get(section, key) for key in config[section]}
    else:
        print(f"Section {section} not found in the {args.conf}")

    # Load training data
    o, d, target_px_values = get_rays(dataset_path, mode='train')

    dataloader = DataLoader(torch.cat((torch.from_numpy(o).reshape(-1, 3).type(torch.float),
                                    torch.from_numpy(d).reshape(-1, 3).type(torch.float),
                                    torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)), dim=1),
                        batch_size=batch_size, shuffle=True)

    dataloader_warmup = DataLoader(torch.cat((torch.from_numpy(o).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                                torch.from_numpy(d).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                                torch.from_numpy(target_px_values).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float)), dim=1),
                        batch_size=batch_size, shuffle=True)

    #test_o, test_d, test_target_px_values = get_rays(dataset_path, mode='test')

    setup_experiment_folders(experiment_name)
    create_experiment_data(experiment_name, test_params) # FIXME: implement this

    for i in range(N):

        model = Nerf(hidden_dim=256).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

        training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, 1, dataloader_warmup, model_name="experiments/"+experiment_name+"/models/M"+str(i)+'.pth', device=device)
        phase = "warmup"
        plt.plot(training_loss)
        filename = "experiments/"+experiment_name+"/figures/model_"+str(i)+"_"+phase+".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
        
        training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, model_name="experiments/"+experiment_name+"/models/M"+str(i)+'.pth', device=device)
        phase = 'training'
        plt.plot(training_loss)
        filename = "experiments/"+experiment_name+"/figures/model_"+str(i)+"_"+phase+".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run NeRF training with specified configuration file.")
    parser.add_argument('--conf', type=str, default="./configs/conf.conf", help="Path to the configuration file")
    args = parser.parse_args()
    main(args)