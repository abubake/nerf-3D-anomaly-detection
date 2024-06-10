import torch
from sklearn.mixture import GaussianMixture
import numpy as np

def compute_accumulated_transmittance(betas):
    accumulated_transmittance = torch.cumprod(betas, 1)     
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
                      accumulated_transmittance[:, :-1]), dim=1)

def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
    
    t = torch.linspace(tn, tf, nb_bins).to(device) # [nb_bins]
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))
    
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) # [nb_rays, nb_bins, 3]    
    
    colors, density = model.intersect(x.reshape(-1, 3), rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    
    colors = colors.reshape((x.shape[0], nb_bins, 3)) # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins))
    
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0)) # [nb_rays, nb_bins, 1]
        
    weights = compute_accumulated_transmittance(1 - alpha) * alpha # [nb_rays, nb_bins]
    
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
        weight_sum = weights.sum(-1) # [nb_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1) # [nb_rays, 3]
    
    return c

@torch.no_grad() 
def render_uncert(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
    
    t = torch.linspace(tn, tf, nb_bins).to(device) # [nb_bins]
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))
    
    x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1) # [nb_rays, nb_bins, 3]    
    
    colors, density = model.intersect(x.reshape(-1, 3), rays_d.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3))
    
    # colors = colors.reshape((x.shape[0], nb_bins, 3)) # [nb_rays, nb_bins, 3]
    density = density.reshape((x.shape[0], nb_bins))
    
    alpha = 1 - torch.exp(- density * delta.unsqueeze(0)) # [nb_rays, nb_bins, 1]
        
    weights = compute_accumulated_transmittance(1 - alpha) * alpha # [nb_rays, nb_bins]
    # beta = 1 - alpha
    # weighted_betas = weights*alpha
    # beta and weights are the same size
    # uncertainty = (weights.unsqueeze(-1) * beta.unsqueeze(-1)).sum(1) # [nb_rays, 1] # where uncertainty betas are just the regular betas
    #uncertainty = entropy(weights, dim=1)
    uncertainty = entropy(weights, dim=1)
    
    return uncertainty, weights, t, alpha, density

# @torch.no_grad() 
# def gaussian_kernel(x, x_i, bandwidth):
#     """Compute the Gaussian kernel between points x and x_i with a given bandwidth."""
#     return torch.exp(-0.5 * ((x - x_i) / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))

# def kde(x, data, bandwidth):
#     """Kernel Density Estimation using the Gaussian kernel."""
#     n = data.size(0)
#     kde_sum = torch.zeros_like(x)
    
#     for i in range(n):
#         kde_sum += gaussian_kernel(x, data[i], bandwidth)
    
#     return kde_sum / n

@torch.no_grad() 
def entropy(val, dim):
    """
    Compute entropy of a probability distribution given by `tensor` along the specified `dim`.
    """
 
    row_sums = val.sum(dim=1, keepdim=True)
    probs = val / row_sums
    
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon) # entropy is base e "nats"
    entropy = -(probs *  log_probs).sum(dim=dim)
    #print(torch.min(entropy)) # prints the entropy of each pixel
    return entropy # returns entropy for each pixel in batch