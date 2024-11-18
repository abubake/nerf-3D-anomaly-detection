from tqdm import tqdm
from rendering import rendering, rendering_uncertainty
import numpy as np
import torch
import time


def loss_function(pred_mean_density, pred_variance, conflict_mask=None, empty_mask=None, 
                  lambda_conflict=1.0, lambda_empty=1.0, lambda_empty_density=1.0):
    """
    Loss function that is based on density, uncertainty, and optional masks for conflict and empty regions.
    """

    total_loss = 0.0  # Start with a total loss of 0

    # Conflict penalty: encourages higher uncertainty in regions with conflicting data
    if conflict_mask is not None:
        conflict_penalty = lambda_conflict * torch.mean(conflict_mask * pred_variance)
        total_loss += conflict_penalty  # Add the conflict penalty to the total loss

    # Empty space penalty: encourages lower uncertainty and density in empty regions
    if empty_mask is not None:
        empty_variance_penalty = lambda_empty * torch.mean(empty_mask * pred_variance)
        empty_density_penalty = lambda_empty_density * torch.mean(empty_mask * pred_mean_density ** 2)
        total_loss += empty_variance_penalty + empty_density_penalty  # Add the empty space penalties to the total loss

    return total_loss


def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, model_name='nerf/torus1.pth', device='cpu'):

    '''
    Default NeRF training function. Takes in the model and all rays at once for offline learning;
    performs supervised learning, comparing predicted images to rendered images for comparison
    
    Returns the training loss, and generates a .pth model file.
    '''
    
    training_loss = []
    start_training_time = time.time()

    for epoch in (range(nb_epochs)):

        epoch_start_time = time.time()

        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            pixel_color_target = batch[:, 6:].to(device)
            
            pixel_color_pred = rendering_uncertainty(model, o, d, tn, tf, nb_bins=nb_bins, device=device)

            t = torch.linspace(tn, tf, nb_bins).to(device) # [nb_bins]
            x = o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * d.unsqueeze(1) # [nb_rays, nb_bins, 3]  
            _, mean_density, variance_density = model.intersect(x.reshape(-1, 3), d.expand(x.shape[0], x.shape[1], 3).transpose(0, 1).reshape(-1, 3))
            
            #loss = ((prediction - target)**2).mean()
            loss = loss_function(mean_density, variance_density)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Final training loss for epoch {epoch} = {training_loss[-1]}")
        print(f"Time for epoch {epoch}: {epoch_duration:.2f} seconds")

        scheduler.step()
        torch.save(model.cpu(), model_name)
        model.to(device)

    total_training_time = time.time() - start_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")
        
    return training_loss


def online_training(model, optimizer, scheduler, tn, tf, nb_bins, cum_training_loss, data_loader, device='cpu'):
    '''
    Online training function for nerf. Incrementally takes in each training image as it is acquired, and updates model. 
    '''
    for batch in tqdm(data_loader):
        o = batch[:, :3].to(device)
        d = batch[:, 3:6].to(device)
        
        target = batch[:, 6:].to(device)
        
        prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
        
        loss = ((prediction - target)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_training_loss.append(loss.item())
        
    scheduler.step()
    
    #torch.save(model.cpu(), model_name)
    #model.to(device)
        
    return cum_training_loss


def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))


def save_checkpoint(model, optimizer, scheduler, filepath='checkpoint.pth'):
    checkpoint = {
        #'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, filepath)
    #print(f"Checkpoint saved at epoch {epoch}")
    print(f"Checkpoint saved")


def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #epoch = checkpoint['epoch']
    print(f"Checkpoint loaded")
    return checkpoint
    

@torch.no_grad()
def test(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None):
    '''
    Renders an image from the test dataset using the trained machine learning model.
    
    In: ...
    Returns: 
        - The rendered image as a variable if target=None
        - The rendered image, mse, and psnr of the image if target/ground truth image is provided
    '''
    
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)
     
    image = []
    for o_batch, d_batch in zip(o, d):
        img_batch = rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
        image.append(img_batch) # N, 3
    image = torch.cat(image)
    image = image.reshape(H, W, 3).cpu().numpy()
    
    if target is not None:
        mse = ((image - target)**2).mean()
        psnr = mse2psnr(mse)
    
    if target is not None: 
        return image, mse, psnr
    else:
        return image


@torch.no_grad()
def test_uncert(model, o, d, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None):
    '''
    Renders an entropy image to test
    '''
    
    o = o.chunk(chunk_size)
    d = d.chunk(chunk_size)
    
    image = []
    for o_batch, d_batch in zip(o, d):
        img_batch, weights, _, _, _ = render_uncert(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
        image.append(img_batch) # N, 3
    image = torch.cat(image)
    image = image.reshape(H, W, 1).cpu().numpy()
    
    if target is not None:
        mse = ((image - target)**2).mean()
        psnr = mse2psnr(mse)
    
    if target is not None: 
        return image, mse, psnr
    else:
        return image