import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import logging
import MDAnalysis as mda

from torch.autograd import Variable
from transvae.structure_prediction import biostructure_to_rmsds

def vae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, self, beta=1):
    "Binary Cross Entropy Loss + Kullback leibler Divergence"
    x = x.long()[:,1:] - 1 #drop the start token
    x = x.contiguous().view(-1) #squeeze into 1 tensor size num_batches*max_seq_len
    x_out = x_out.contiguous().view(-1, x_out.size(2)) # squeeze first and second dims matching above, keeping the 25 class dims.
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)  #smiles strings have 25 classes or characters (check len(weights))

    if pred_prop is not None:
        if "decision_tree" in self.params["type_pp"]:            
            bce_prop=torch.tensor(0.)
        else: 
            # check prediction types. 
            # when None, assume binary classification
            # when not None, loop through each prediction type
            if (self.params["prediction_types"] is None):
                bce_prop = F.binary_cross_entropy(
                    pred_prop.squeeze(-1)[~torch.isnan(true_prop)], 
                    true_prop[            ~torch.isnan(true_prop)]
                )
            else:
                prop_losses = []
                for i in range(pred_prop.shape[1]):
                    if self.params["prediction_types"][i] == "classification":
                        _prop_loss = F.cross_entropy(
                            pred_prop[:,i][~torch.isnan(true_prop[:,i])], 
                            true_prop[:,i][~torch.isnan(true_prop[:,i])]
                        )
                        prop_losses.append( _prop_loss )
                    else:
                        _prop_loss = F.mse_loss(
                            pred_prop[:,i][~torch.isnan(true_prop[:,i])], 
                            true_prop[:,i][~torch.isnan(true_prop[:,i])]
                        )
                        prop_losses.append( _prop_loss )
                bce_prop = torch.sum(torch.stack(prop_losses))
            #bce_prop = F.cross_entropy(pred_prop.squeeze(-1)[~torch.isnan(true_prop)], true_prop[~torch.isnan(true_prop)])
    else:
        bce_prop = torch.tensor(0.)
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCE + KLD + bce_prop, BCE, KLD, bce_prop

def trans_vae_loss(x, x_out, mu, logvar, true_len, pred_len, true_prop, pred_prop, weights, self, beta=1,beta_property=1):
    "Binary Cross Entropy Loss + Kullbach leibler Divergence + Mask Length Prediction"
    batch_size, _ = x.size()
    x = x.long()[:,1:] - 1
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2))
    true_len = true_len.contiguous().view(-1)
    BCEmol  = F.cross_entropy(x_out, x,           reduction='mean', weight=weights)
    BCEmask = F.cross_entropy(pred_len, true_len, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    if pred_prop is not None:
        if "decision_tree" in self.params["type_pp"]:
            print(pred_prop)
        else: 
            # check prediction types. 
            # when None, assume binary classification
            # when not None, loop through each prediction type
            if (self.params["prediction_types"] is None):
                bce_prop = F.binary_cross_entropy(
                    pred_prop.squeeze(-1)[~torch.isnan(true_prop)], 
                    true_prop[            ~torch.isnan(true_prop)]
                )
            else:
                _nan_encountered = False
                prop_losses = []
                for i in range(pred_prop.shape[1]):
                    if self.params["prediction_types"][i] == "classification":
                        _prop_loss = F.cross_entropy(
                            pred_prop[:,i][~torch.isnan(true_prop[:,i])], 
                            true_prop[:,i][~torch.isnan(true_prop[:,i])]
                        )
                    else:
                        _prop_loss = F.mse_loss(
                            pred_prop[:,i][~torch.isnan(true_prop[:,i])], 
                            true_prop[:,i][~torch.isnan(true_prop[:,i])]
                        )
                    
                    if len( true_prop[:,i][~torch.isnan(true_prop[:,i])] ) == 0:
                        prop_losses.append( torch.tensor(0.) )        
                    else:
                        prop_losses.append( _prop_loss )
                bce_prop = torch.sum(torch.stack(prop_losses))
            
            bce_prop = beta_property * bce_prop
            #bce_prop = F.binary_cross_entropy(pred_prop.squeeze(-1), true_prop)
            # bce_prop = F.cross_entropy(pred_prop.squeeze(-1), true_prop)
    else:
        bce_prop = torch.tensor(0.)
        
    if torch.isnan(KLD):
        KLD = torch.tensor(0.)
    return BCEmol + BCEmask + KLD + bce_prop, BCEmol, BCEmask, KLD, bce_prop

def aae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, self, latent_codes, opt, train_test, beta=1):
    #formatting x
    x = x.long()[:,1:] - 1 
    x = x.contiguous().view(-1) 
    x_out = x_out.contiguous().view(-1, x_out.size(2))

    #generator and autoencoder loss
    opt.g_opt.zero_grad() #zeroing gradients
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights) #autoencoder loss
    if 'gpu' in self.params['HARDWARE']:
        valid_discriminator_targets =  Variable(torch.ones(latent_codes.shape[0], 1), requires_grad=False).cuda() #valid
    else:
        valid_discriminator_targets =  Variable(torch.ones(latent_codes.shape[0], 1), requires_grad=False) #valid
        
    generator_loss = F.binary_cross_entropy_with_logits(self.discriminator(latent_codes),
                                                        valid_discriminator_targets) #discriminator loss vs. valid
    
    if pred_prop is not None:
        if "decision_tree" in self.params["type_pp"]:
            print(pred_prop)
        else: 
            bce_prop = F.binary_cross_entropy(pred_prop.squeeze(-1), true_prop)
    else:
        bce_prop = torch.tensor(0.)
    auto_and_gen_loss = BCE + generator_loss + bce_prop
    if 'train' in train_test:
        auto_and_gen_loss.backward() #backpropagating generator
        opt.g_opt.step()
        
    #discriminator loss: discriminator's ability to classify real from generated samples
    opt.d_opt.zero_grad()#zeroing gradients
    if 'gpu' in self.params['HARDWARE']:
        fake_discriminator_targets = Variable(torch.zeros(latent_codes.shape[0],1), requires_grad=False).cuda() #fake
    else:
        fake_discriminator_targets = Variable(torch.zeros(latent_codes.shape[0],1), requires_grad=False) #fake
    fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(latent_codes.detach()),fake_discriminator_targets)
    
    if 'gpu' in self.params['HARDWARE']:
        discriminator_targets = Variable(torch.ones(latent_codes.shape[0],1), requires_grad=False).cuda() #valid
        noise = Variable(torch.Tensor(np.random.normal(0,1,(latent_codes.shape[0],1))), requires_grad=False).cuda() #fake
    else:
        discriminator_targets = Variable(torch.ones(latent_codes.shape[0],1), requires_grad=False) #valid
        noise = Variable(torch.Tensor(np.random.normal(0,1,(latent_codes.shape[0],1))), requires_grad=False) #fake
    real_loss = F.binary_cross_entropy_with_logits(noise,discriminator_targets)
        
    disc_loss = 0.5*fake_loss + 0.5*real_loss
        
    total_loss = disc_loss
    
    if 'train' in train_test:
        total_loss.backward() #backpropagating
        opt.d_opt.step() 
   
    return auto_and_gen_loss, BCE, torch.tensor(0.), bce_prop, disc_loss

def wae_loss(x, x_out, mu, logvar, true_prop, pred_prop, weights, latent_codes, self, beta=1):
    "reconstruction and mmd loss"
    #reconstruction loss
    x = x.long()[:,1:] - 1 
    x = x.contiguous().view(-1)
    x_out = x_out.contiguous().view(-1, x_out.size(2)) 
    BCE = F.cross_entropy(x_out, x, reduction='mean', weight=weights)  #smiles strings have 25 classes or characters (check len(weights))
    
   
    z_tilde = latent_codes
    z_var = 2 #variance of gaussian
    sigma = math.sqrt(2)#sigma (Number): scalar variance of isotropic gaussian prior P(Z). set to sqrt(2)
    if 'gpu' in self.params['HARDWARE']:
        z = sigma*torch.randn(latent_codes.shape).cuda() #sample gaussian
    else:
        z = sigma*torch.randn(latent_codes.shape) #sample gaussian
    n = z.size(0)
    im_kernel_sum_1 = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1))
    im_kernel_sum_2 = im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1))
    im_kernel_sum_3 = -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)
    
    mmd =  im_kernel_sum_1+im_kernel_sum_2+im_kernel_sum_3
    
    if pred_prop is not None:
        if "decision_tree" in self.params["type_pp"]:
            print(pred_prop)
        else: 
            bce_prop = F.binary_cross_entropy(pred_prop.squeeze(-1), true_prop)
    else:
        bce_prop = torch.tensor(0.)
    
    return BCE + mmd + bce_prop, BCE, torch.tensor(0.), bce_prop, mmd

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    "adapted from  https://github.com/1Konny/WAE-pytorch/blob/master/ops.py"
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum

def deep_isometry_loss(mu, sequences, pairwise_distances, beta=None, reduction='mean'):
    """
    Deep Isometry Loss. 
    Computes the difference in distance b/w latent space points 
    and their corresponding inputted points' aligned rmsds. 

    Parameters:
    ----------
        mu : torch.tensor
             latent space points
        target_distances : torch.tensor 
                desired distances between latent space points

    Returns:
    -------
        loss : torch.tensor
               difference in distance b/w latent space points 
               and their corresponding target distance
    """
    _n = len(mu)
    target_distances = torch.zeros((_n*(_n-1))//2, dtype=torch.float32)
    _dist_idx = 0
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            target_distances[_dist_idx] = pairwise_distances.get(sequences[i]+"_"+sequences[j],-1)
            _dist_idx += 1

    # compute pairwise distances in the latent space using mu_subset
    # https://pytorch.org/docs/stable/generated/torch.cdist.html
    logging.info("computing pairwise distances between latent space points")
    _n = len(mu)
    _mu_pairwise_distances = torch.zeros((_n*(_n-1))//2, dtype=torch.float32)
    _mu_idx = 0
    for i in range(_n):
        for j in range(i+1, _n):
            _mu_pairwise_distances[_mu_idx] = torch.dist(mu[i], mu[j], p=2)
            _mu_idx += 1

    if len(target_distances)!=len(_mu_pairwise_distances):
        raise ValueError(f"Number of pairwise distances ({len(_mu_pairwise_distances)}) and target distances ({len(target_distances)}) do not match")

    # ignore indices where target_distances = -1
    _mu_pairwise_distances = _mu_pairwise_distances[target_distances!=-1]
    target_distances = target_distances[target_distances!=-1]

    logging.info(f"Number of pairwise distances: {_mu_pairwise_distances.shape[0]}")
    if _mu_pairwise_distances.shape[0] > 0:
        logging.info(f"min, max pairwise distances: {_mu_pairwise_distances.min()}, {_mu_pairwise_distances.max()}")

    # compute difference between pairwise distances and rmsds
    _diff  = torch.pow(_mu_pairwise_distances - target_distances,2) # basically |d(z_i, z_j) - d(x_i, x_j)|
    if reduction == 'mean':
        loss = torch.mean(_diff)
    else:
        loss = _diff

    if beta is None:
        return loss
    else:
        return beta*loss

def triplet_loss():
    pass