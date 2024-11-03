import comet_ml

import os
import json
import sys
import logging
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *
from transvae.opt import NoamOpt
from transvae.data import vae_data_gen, make_std_mask
from transvae.loss import vae_loss, trans_vae_loss, aae_loss, wae_loss, deep_isometry_loss
from transvae.DDP import reduce_tensor

from pytorch_metric_learning import losses as ptlosses

import torch.distributed as dist
import torch.utils.data.distributed


####### VAE SHELL ##########

class VAEShell():
    """
    VAE shell is a parent class to any created VAE architecture
    takes care of general functions e.g. saving, loading, training, property predictor
    """
    def __init__(self, params, name=None):
        self.params = params
        self.name = name
        #Hardware and DDP need to be given before being able to load a model
        if 'HARDWARE' not in self.params.keys():
            self.params['HARDWARE'] = 'gpu'
        if 'DDP' not in self.params.keys():
            self.params['DDP'] = False
        #The character dictionnary is shared for all models
        if 'CHAR_DICT' in self.params.keys():
            self.vocab_size = len(self.params['CHAR_DICT'].keys())
            self.pad_idx = self.params['CHAR_DICT']['_']
            if 'CHAR_WEIGHTS' in self.params.keys():
                self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
            else:
                self.params['CHAR_WEIGHTS'] = torch.ones(self.vocab_size, dtype=torch.float)
                
        ### Sequence length hard-coded into model
        self.src_len = 126
        self.tgt_len = 125

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'params': self.params}
        self.loaded_from = None

    def save(self, state, fn, path='checkpointz', use_name=True):
        """
        Saves current model state to .ckpt file
        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        if self.params['DDP']:
            torch.save(state, save_path)
            os.system("cp {} ~/scratch".format(save_path))
        else:
            torch.save(state, save_path)

        # also save beta annealer state
        
    def load(self, checkpoint_path, rank=0, workaround=None):
        """
        Loads a saved model state
        Arguments:
            checkpoint_path (str, required): Path to saved .ckpt file
        """
        ###Loading on either GPU or CPU 
        if workaround is not None:
            self.params["HARDWARE"] = workaround
        if 'gpu' in self.params['HARDWARE']:
            if self.params['DDP']:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                loaded_checkpoint = torch.load(checkpoint_path, map_location=map_location)
            else:
                loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        else: 
            loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        ###Load checkpoint from specified path and saved state
        self.loaded_from = checkpoint_path
        ###This is the outer dict defined on line 47 above
        for k in self.current_state.keys():
            try: 
                self.current_state[k] = loaded_checkpoint[k]
            except KeyError: 
                self.current_state[k] = None

        self.vocab_size = len(self.current_state['params']['CHAR_DICT'].keys())
        self.pad_idx   = self.current_state['params']['CHAR_DICT']['_']
        state_dict     = self.current_state['model_state_dict']
        self.name      = self.current_state['name']
        self.n_epochs  = self.current_state['epoch']
        self.best_loss = self.current_state['best_loss']
        #This is the last key in the outer dict. Need to match the values from the ckpt.
        for k, v in self.current_state['params'].items():
            if k in self.arch_params or k not in self.params.keys():
                self.params[k] = v
        #once we have all the parameters set we build the model        
        self.build_model()    
        
        # Necessary code to remove additional 'module' string attached to 'dict_items' by DDP model
        if 'module' in list(state_dict)[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(self.current_state['model_state_dict'])        
        
        #Load the aae generator and discriminator optimizers separately
        if self.model_type == 'aae': 
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'][0],self.current_state['optimizer_state_dict'][1])
        else:
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
            
    def train(self, train_mols, val_mols, train_props=None, val_props=None,
              epochs=200, use_contrastive_loss=False, pairwise_distances=None, inputs_w_distances=None,
              save=True, save_freq=None, log=True, log_dir='trials', comet_experiment=None):
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
            train_props (np.array): Numpy array containing chemical property of
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
            use_contrastive_loss: (bool) If true, uses contrastive loss,

        Notes:
        - use_contrastive_loss, the arguments being provided to it are call use_isometry_loss due to historical testing of an isometry loss method.
        """
        torch.backends.cudnn.benchmark = True #optimize run-time for fixed model input size
        structure_predictor = None # DEPRECATED
        ### Prepare data iterators
        train_data, binners = vae_data_gen(train_mols, self.src_len, self.name, train_props, 
                                  char_dict=self.params['CHAR_DICT'], 
                                  mask_label_percent=self.params['mask_label_percent'], use_contrastive_loss=use_contrastive_loss
        )
        val_data, _   = vae_data_gen(  val_mols, self.src_len, self.name,   val_props, 
                                  char_dict=self.params['CHAR_DICT'], 
                                  mask_label_percent=self.params['mask_label_percent'],
                                  binners=binners, use_contrastive_loss=use_contrastive_loss
                                  )
        
        # special input for isometry learning
        use_isometry_loss=False
        if use_isometry_loss:
            logging.info(f"generating encoded sequences for isometry loss")
            assert inputs_w_distances is not None, "ERROR: Must provide inputs with distances"
            train_data_w_distances = vae_data_gen(inputs_w_distances[0].values, self.src_len, self.name, None, char_dict=self.params['CHAR_DICT'])
            val_data_w_distances   = vae_data_gen(inputs_w_distances[1].values, self.src_len, self.name, None, char_dict=self.params['CHAR_DICT'])

        #SPECIAL DATA INPUT FOR DDP
        if self.params['DDP']:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(  val_data, shuffle=True)
            train_iter  = torch.utils.data.DataLoader(train_data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                num_workers=0,
                                                pin_memory=False, drop_last=True, sampler=train_sampler)
            val_iter    = torch.utils.data.DataLoader(val_data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                num_workers=0,
                                                pin_memory=False, drop_last=True, sampler=val_sampler)
        else:
            train_iter  = torch.utils.data.DataLoader(train_data,
                                                 batch_size=self.params['BATCH_SIZE'],
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=False, drop_last=True)
            val_iter    = torch.utils.data.DataLoader(val_data,
                                               batch_size=self.params['BATCH_SIZE'],
                                               shuffle=True, num_workers=0,
                                               pin_memory=False, drop_last=True)
        
        self.chunk_size = self.params['BATCH_SIZE'] // self.params['BATCH_CHUNKS'] #chunks for TransVAE

      
        ### If save freq not specified set to num epochs
        if save_freq is None:
            save_freq = epochs
            
        ### Setup log file
        if log:
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
            except FileNotFoundError:
                already_wrote = False
            log_file = open(log_fn, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,'\
                               'kld_loss,prop_bce_loss,disc_loss,mmd_loss,isometry_loss,run_time\n')
            log_file.close()

        ### Contrastive loss
        contrastive_loss = ptlosses.SupConLoss() # supervised contrastive loss

        ### Initialize Annealer
        kl_annealer = KLAnnealer(self.params['BETA_INIT'], self.params['BETA'],
                                 epochs, self.params['ANNEAL_START'])
        
        contrastive_kl_start = 0.01
        property_init = 0.0
        property_final= 100.0 if use_contrastive_loss is False else 0.0
        _total_epochs = epochs
        if self.n_epochs>0:
            _total_epochs = self.n_epochs + epochs
            _m = (1.0-contrastive_kl_start)/(_total_epochs-self.params["ANNEAL_START"])
            contrastive_kl_start = _m*self.n_epochs + contrastive_kl_start

            property_init = ((property_final-property_init)/(_total_epochs))*self.n_epochs
        contrastive_annealer = KLAnnealer(contrastive_kl_start, 1.0,
                                          _total_epochs, self.params['ANNEAL_START'])
        
        property_annealer = KLAnnealer(property_init, property_final, epochs, 0)
        ####################################################################################################
        ### Epoch loop start
        for epoch in range(epochs):
            if self.params['DDP']: #Synchronize the GPUs for checkpoint loading using DDP
                ngpus_per_node = torch.cuda.device_count()
                local_rank = int(os.environ.get("SLURM_LOCALID")) 
                rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
                if rank==0 or rank==1 or rank==2 or rank==3:
                    dist.barrier()
        
            epoch_start_time= perf_counter()

            # linearly ramp up the isometry loss
            isometry_loss_weighting = 1.0#min(epochs//2, epoch*(epochs//2) ) 

            ##################################
            ### Train Loop
            ##################################
            self.model.train()
            losses = []
            recon_losses = []
            prop_losses  = []
            kld_losses   = []
            contrastive_losses = []
            if self.loaded_from is not None:
                # +1 because epoch is 0-indexed, and we don't want to repeat the last
                # beta value from the previous training session
                beta = kl_annealer(epoch+1)
                beta_contrastive = contrastive_annealer(epoch+1)
                beta_property = property_annealer(epoch+1+self.n_epochs)
            else:
                beta = kl_annealer(epoch)
                beta_contrastive = contrastive_annealer(epoch)
                beta_property = property_annealer(epoch+self.n_epochs)

            for j, data in enumerate(train_iter):
                avg_losses          = []
                avg_bce_losses      = []
                avg_bcemask_losses  = []
                avg_kld_losses      = []
                avg_prop_bce_losses = []
                avg_disc_losses     = []
                avg_mmd_losses      = []
                avg_isometry_losses     = []
                avg_contrastive_losses = []
                start_run_time = perf_counter()
                
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data  = batch_data[:,                        :-self.params["d_pp_out"]]
                    props_data = batch_data[:,-self.params["d_pp_out"]:                        ]

                    # sample sequences w distances
                    if use_isometry_loss:
                        n_to_grab = max(5, int(self.chunk_size*0.1) ) # grab 10% of the batch, or 5
                        idxs = np.random.choice(len(train_data_w_distances), n_to_grab, replace=False)
                        batch_data_w_distances = np.take(
                            train_data_w_distances, idxs, axis=0
                        )
                        mols_data_w_distances = batch_data_w_distances[:, :-1]

                        # now replace some of the sequences with the ones with distances
                        n_to_replace = n_to_grab
                        replace_idxs = np.random.choice(self.chunk_size, n_to_replace, replace=False)
                        mols_data[replace_idxs] = mols_data_w_distances
                        logging.info(f"grabbed {n_to_grab} sequences with distances")

                    #Move data to GPU if available
                    if 'gpu' in self.params['HARDWARE']:
                        mols_data  =  mols_data.cuda()
                        props_data = props_data.cuda()
                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()                 
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2) #true or false according to sequence length
                    tgt_mask = make_std_mask(tgt, self.pad_idx) #cascading true false masking [true false...] [true true false...] ...
                        
                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_bce = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'], self,
                                                                            beta, beta_property=beta_property)
                        if use_contrastive_loss:
                            if true_prop.shape[1] == 1:
                                _normalized_mu = mu / torch.linalg.vector_norm(mu, dim=1, ord=2, keepdim=True)
                                _contrastive_loss = contrastive_loss(_normalized_mu, true_prop.flatten())
                            else:
                                error_msg = f"contrastive loss only supported for single property, not {true_prop.shape=}"
                                raise ValueError(error_msg)
                            loss = loss + _contrastive_loss
                        else:
                            _contrastive_loss = torch.tensor(0.0)

                        avg_bcemask_losses.append(bce_mask.item())
                        
                    if self.model_type == 'aae': #the aae loss is calculated in the forward function due to the 2 backward passes
                        loss, bce, kld, prop_bce, disc_loss = self.model(src, tgt, true_prop,self.params['CHAR_WEIGHTS'], beta,
                                                                              self.optimizer, 'train', src_mask, tgt_mask)
                        avg_disc_losses.append(disc_loss.item()) #append the disc loss from aae
                        
                    if self.model_type == 'wae': 
                        x_out, mu, logvar, pred_prop, latent_mem = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        loss, bce, kld, prop_bce, mmd_loss  = wae_loss(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  latent_mem, self,
                                                                  beta)
                        avg_mmd_losses.append(mmd_loss.item()) #append the mmd loss from wae
                        
                    if self.model_type == 'rnn' or self.model_type =='rnn_attn':
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        #with open("trials/property_predictor_output.txt","a") as fpp:
                        #    fpp.write(f"pred_prop = {pred_prop}\n")
                        loss, bce, kld, prop_bce = vae_loss(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'], self,
                                                                  beta)

                    if use_isometry_loss:
                        if "peptide" not in self.name:
                            raise ValueError("isometry loss only supported for peptide models")
                        _total_n_pairs_to_use = len(mu) # total number might be large, so use a random subset
                        # N choose 2 = N! / 2!(N-2)! = N(N-1)/2 = M
                        # N^2 - N - 2M = 0
                        # N = (1 + sqrt(1 + 8M))/2
                        _choose_n = (1 + np.sqrt(1 + 8*_total_n_pairs_to_use))/2
                        _choose_n = int(_choose_n)
                        _idx = np.random.choice(len(mu), _choose_n, replace=False)
                        _idx = replace_idxs # use the same indices as above
                        ######################
                        # get distance_targets
                        # must decode sequences in batch
                        ######################
                        _sequences = mols_data[_idx]
                        _sequence_subset = decode_seq(_sequences, self.params['CHAR_DICT'])

                        # latent points
                        mu_subset = mu[_idx]
                        
                        isometry_loss = deep_isometry_loss(mu_subset, _sequence_subset, pairwise_distances, beta=beta_contrastive)
                        # isometry_loss = deep_isometry_loss(mu_subset, _sequence_subset, pairwise_distances)
                        # increase the total loss by the rmsd loss
                        loss = loss + isometry_loss_weighting*isometry_loss
                    else:
                        isometry_loss = torch.tensor(0.0)
                    
                    # perform all reduce if distributed training
                    if self.params['DDP']:
                        loss_reduced = reduce_tensor(loss)
                        bce          = reduce_tensor(bce)
                        kld          = reduce_tensor(kld)
                        if prop_bce.device.type == 'cuda':
                            prop_bce = reduce_tensor(prop_bce)

                        if use_isometry_loss:
                            isometry_loss = reduce_tensor(isometry_loss)
                        else:
                            isometry_loss = torch.tensor(0.0)

                        if use_contrastive_loss:
                            _contrastive_loss = reduce_tensor(_contrastive_loss)
                        else:
                            _contrastive_loss = torch.tensor(0.0)
                    
                        avg_losses.append(  loss_reduced.item())
                    else:
                        avg_losses.append(          loss.item())
                    avg_bce_losses.append(           bce.item())
                    avg_kld_losses.append(           kld.item())
                    avg_prop_bce_losses.append( prop_bce.item())
                    avg_isometry_losses.append(    isometry_loss.item())
                    avg_contrastive_losses.append( _contrastive_loss.item())

                    if not self.model_type == 'aae': #the aae backpropagates in the loss function
                        loss.backward()
                
                if not self.model_type == 'aae':
                    self.optimizer.step()
                    disc_loss = 0 
                    self.model.zero_grad()
                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(    avg_losses)
                avg_bce  = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                if len(avg_disc_losses) == 0:
                    avg_disc = 0
                else:
                    avg_disc = np.mean(avg_disc_losses)
                if len(avg_mmd_losses) == 0:
                    avg_mmd = 0
                else:
                    avg_mmd = np.mean(avg_mmd_losses)

                avg_kld      = np.mean(avg_kld_losses)
                avg_prop_bce = np.mean(avg_prop_bce_losses)
                # avg_rmsd     = np.mean(avg_isometry_losses)
                avg_rmsd     = np.mean(avg_contrastive_losses)
                
                losses.append(avg_loss)
                recon_losses.append(avg_bce)
                prop_losses.append(avg_prop_bce)
                kld_losses.append(avg_kld)
                contrastive_losses.append(avg_rmsd)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'train',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_bcemask,
                                                                         avg_kld,
                                                                         avg_prop_bce,
                                                                         avg_disc,
                                                                         avg_mmd,
                                                                         avg_rmsd,
                                                                         run_time))
                    log_file.close()
            train_loss      = np.mean(losses)
            train_bce_loss  = np.mean(recon_losses)
            train_prop_loss = np.mean(prop_losses)
            train_kld_loss  = np.mean(kld_losses)
            train_rmsd_loss = np.mean(contrastive_losses)
            
            
            ##############################
            ### Validation Loop
            ##############################
            self.model.eval()
            losses = []
            recon_losses = []
            prop_losses  = []
            kld_losses   = []
            contrastive_losses = []   
            for j, data in enumerate(val_iter):
                avg_losses          = []
                avg_bce_losses      = []
                avg_bcemask_losses  = []
                avg_kld_losses      = []
                avg_prop_bce_losses = []
                avg_disc_losses     = []
                avg_mmd_losses      = []
                avg_isometry_losses     = []
                avg_contrastive_losses = []

                start_run_time = perf_counter()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data  = batch_data[:,                        :-self.params["d_pp_out"]]
                    props_data = batch_data[:,-self.params["d_pp_out"]:                        ]
                    
                    # sample sequences w distances
                    if use_isometry_loss:
                        n_to_grab = max(5, int(self.chunk_size*0.1) ) # grab 10% of the batch, or 5
                        idxs = np.random.choice(len(val_data_w_distances), n_to_grab, replace=False)
                        batch_data_w_distances = np.take(
                            val_data_w_distances, idxs, axis=0
                        )
                        mols_data_w_distances = batch_data_w_distances[:, :-1]

                        # now replace some of the sequences with the ones with distances
                        n_to_replace = n_to_grab
                        replace_idxs = np.random.choice(self.chunk_size, n_to_replace, replace=False)
                        mols_data[replace_idxs] = mols_data_w_distances

                    # move to GPU if available
                    if 'gpu' in self.params['HARDWARE']:
                        mols_data = mols_data.cuda()
                        props_data = props_data.cuda()
                    src = Variable(mols_data).long()
                    tgt = Variable(mols_data[:,:-1]).long()
                    true_prop = Variable(props_data)
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)
                    scores = Variable(data[:,-1])
                  
                    if self.model_type == 'transformer':
                        x_out, mu, logvar, pred_len, pred_prop = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_bce = trans_vae_loss(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.params['CHAR_WEIGHTS'], self,
                                                                            beta, beta_property=beta_property)
                        avg_bcemask_losses.append(bce_mask.item())

                        if use_contrastive_loss:
                            # _contrastive_loss = contrastive_loss(mu, true_prop)
                            if true_prop.shape[1] == 1:
                                _normalized_mu = mu / torch.linalg.vector_norm(mu, dim=1, ord=2, keepdim=True)
                                _contrastive_loss = contrastive_loss(_normalized_mu, true_prop.flatten())
                            else:
                                error_msg = f"contrastive loss only supported for single property, not {true_prop.shape=}"
                                raise ValueError(error_msg)
                            loss = loss + _contrastive_loss
                        else:
                            _contrastive_loss = torch.tensor(0.0)
                        
                    if self.model_type == 'aae':
                        loss, bce, kld, prop_bce, disc_loss = self.model(src, tgt, true_prop,self.params['CHAR_WEIGHTS'], beta,
                                                                         self.optimizer, 'test', src_mask, tgt_mask)
                 
                        avg_disc_losses.append(disc_loss.item()) #added the disc loss from aae
                        
                    if self.model_type == 'wae': 
                        x_out, mu, logvar, pred_prop, latent_mem = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        loss, bce, kld, prop_bce, mmd_loss = wae_loss(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'],
                                                                  latent_mem, self,
                                                                  beta)
                        avg_mmd_losses.append(mmd_loss.item())
                        
                    if self.model_type == 'rnn' or self.model_type =='rnnattn':
                        x_out, mu, logvar, pred_prop = self.model(src, tgt, true_prop, src_mask, tgt_mask)
                        loss, bce, kld, prop_bce = vae_loss(src, x_out, mu, logvar,
                                                                  true_prop, pred_prop,
                                                                  self.params['CHAR_WEIGHTS'], self,
                                                                  beta)
                    
                    if use_isometry_loss:
                        if "peptide" not in self.name:
                            raise ValueError("Structure loss only supported for peptide models")
                        _total_n_pairs_to_use = len(mu) # total number might be large, so use a random subset
                        # N choose 2 = N! / 2!(N-2)! = N(N-1)/2 = M
                        # N^2 - N - 2M = 0
                        # N = (1 + sqrt(1 + 8M))/2
                        _choose_n = (1 + np.sqrt(1 + 8*_total_n_pairs_to_use))/2
                        _choose_n = int(_choose_n)
                        _idx = np.random.choice(len(mu), _choose_n, replace=False)
                        _idx = replace_idxs # use the same indices as above
                        ######################
                        # get distance_targets
                        # must decode sequences in batch
                        ######################
                        _sequences = mols_data[_idx]
                        _sequence_subset = decode_seq(_sequences, self.params['CHAR_DICT'])

                        # latent points
                        mu_subset = mu[_idx]
                        
                        isometry_loss = deep_isometry_loss(mu_subset, _sequence_subset, pairwise_distances,beta=beta_contrastive)
                        # isometry_loss = deep_isometry_loss(mu_subset, _sequence_subset, pairwise_distances)
                        # increase the total loss by the rmsd loss
                        loss = loss + isometry_loss
                    else:
                        isometry_loss = torch.tensor(0.0)

                    # perform all reduce if distributed training
                    if self.params['DDP']:
                        loss_reduced = reduce_tensor(loss)
                        bce          = reduce_tensor(bce)
                        kld          = reduce_tensor(kld)
                        if prop_bce.device.type == 'cuda':
                            prop_bce = reduce_tensor(prop_bce)
                        
                        if use_isometry_loss:
                            isometry_loss = reduce_tensor(isometry_loss)
                        else:
                            isometry_loss = torch.tensor(0.0)
                        
                        if use_contrastive_loss:
                            _contrastive_loss = reduce_tensor(_contrastive_loss)
                        else:
                            _contrastive_loss = torch.tensor(0.0)

                        avg_losses.append(     loss_reduced.item())
                    else:
                        avg_losses.append(         loss.item())
                    avg_bce_losses.append(          bce.item())
                    avg_kld_losses.append(          kld.item())
                    avg_prop_bce_losses.append(prop_bce.item())
                    avg_isometry_losses.append(   isometry_loss.item())
                    avg_contrastive_losses.append(_contrastive_loss.item())

                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(    avg_losses)
                avg_bce  = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                if len(avg_disc_losses) == 0:
                    avg_disc = 0
                else:
                    avg_disc = np.mean(avg_disc_losses)
                if len(avg_mmd_losses) == 0:
                    avg_mmd = 0
                else:
                    avg_mmd = np.mean(avg_mmd_losses)
                
                avg_kld      = np.mean(avg_kld_losses)
                avg_prop_bce = np.mean(avg_prop_bce_losses)
                # avg_rmsd     = np.mean(avg_isometry_losses)
                avg_rmsd     = np.mean(avg_contrastive_losses)

                losses.append(avg_loss)
                recon_losses.append(avg_bce)
                prop_losses.append(avg_prop_bce)
                kld_losses.append(avg_kld)
                contrastive_losses.append(avg_rmsd)   
                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                j, 'test',
                                                                avg_loss,
                                                                avg_bce,
                                                                avg_bcemask,
                                                                avg_kld,
                                                                avg_prop_bce,
                                                                avg_disc,
                                                                avg_mmd,
                                                                avg_rmsd,
                                                                run_time))
                    log_file.close()

            self.n_epochs += 1
            val_loss      = np.mean(            losses)
            val_bce_loss  = np.mean(      recon_losses)
            val_prop_loss = np.mean(       prop_losses)
            val_kld_loss  = np.mean(        kld_losses)
            val_rmsd_loss = np.mean(contrastive_losses)
            epoch_end_time = perf_counter()
            epoch_time = round(epoch_start_time - epoch_end_time, 5)
            if self.params['DDP']:
                os.system("echo from rank {} Epoch - {} Train - {} Val - {} KLBeta - {} Epoch time - {}".format(rank, self.n_epochs, train_loss, val_loss, beta, epoch_time))

                if (comet_experiment is not None) and (rank==0):
                    _comet_package = {
                        "train_loss": train_loss,
                        "train_reconstruction_loss": train_bce_loss,
                        "train_kld_loss": train_kld_loss,
                        "train_property_loss": train_prop_loss,
                        "train_contrastive_loss": train_rmsd_loss,

                        "val_loss": val_loss,
                        "val_reconstruction_loss": val_bce_loss,
                        "val_kld_loss": val_kld_loss,
                        "val_property_loss": val_prop_loss,
                        "val_contrastive_loss": val_rmsd_loss,

                        "kl_beta": beta,
                        "prop_beta": beta_property,
                        "epoch_time": epoch_time
                    }
                    comet_experiment.log_metrics(_comet_package, step=self.n_epochs)
            else:
                print('Epoch - {} Train - {} Val - {} KLBeta - {} Epoch time - {}'.format(self.n_epochs, train_loss, val_loss, beta, epoch_time))
                
                if comet_experiment is not None:
                    _comet_package = {
                        "train_loss": train_loss,
                        "train_reconstruction_loss": train_bce_loss,
                        "train_kld_loss": train_kld_loss,
                        "train_property_loss": train_prop_loss,
                        "train_contrastive_loss": train_rmsd_loss,

                        "val_loss": val_loss,
                        "val_reconstruction_loss": val_bce_loss,
                        "val_kld_loss": val_kld_loss,
                        "val_property_loss": val_prop_loss,
                        "val_contrastive_loss": val_rmsd_loss,

                        "kl_beta": beta,
                        "prop_beta": beta_property,
                        "epoch_time": epoch_time
                    }
                    comet_experiment.log_metrics(_comet_package, step=self.n_epochs)

            ### check if any loss has NaNed out, system exit with message and error code
            if np.isnan(train_loss) or np.isnan(val_loss):
                sys.exit("Loss is NaN, exiting")
            # also check if inf
            if np.isinf(train_loss) or np.isinf(val_loss):
                sys.exit("Loss is inf, exiting")
            ### Update current state and save model
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            if self.model_type == 'aae': #The aae uses two optimizers we store both optimizer states in a tuple, see load for loading
                self.current_state['optimizer_state_dict'] = (self.optimizer.state_dict_g,self.optimizer.state_dict_d)
            else:
                self.current_state['optimizer_state_dict'] = self.optimizer.state_dict
           
            if (self.n_epochs) % save_freq == 0:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:                
                    if self.params['DDP']:
                        if rank ==0:
                            self.save(self.current_state, epoch_str,path=self.params['save_dir'])
                    else: 
                        self.save(self.current_state, epoch_str,path=self.params['save_dir'])

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.params['d_latent'])
        else:
            assert sample_dims is not None, "ERROR: Must provide sample dimensions"
            if mode == 'top_dims':
                z = torch.zeros((size, self.params['d_latent']))
                for d in sample_dims:
                    z[:,d] = torch.randn(size)
            elif mode == 'k_dims':
                z = torch.zeros((size, self.params['d_latent']))
                d_select = np.random.choice(sample_dims, size=k, replace=False)
                for d in d_select:
                    z[:,d] = torch.randn(size)
        return z

    def greedy_decode(self, mem, print_step=100 ,src_mask=None, return_probabilities=False, verbose=False):
        """
        Greedy decode from model memory if the model is a transformer
        Otherwise just decode from memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        start_symbol = self.params['CHAR_DICT']['<start>']
        max_len = self.tgt_len
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long()
        tgt = torch.ones(mem.shape[0],max_len+1).fill_(start_symbol).long()
        
        if return_probabilities:
            probabilities = torch.ones((mem.shape[0], max_len, self.vocab_size-1)) # -1 to remove predicting start token 

        if src_mask is None and self.model_type == 'transformer':
            mask_lens = self.model.encoder.predict_mask_length(mem)
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len+1))
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len))
        
        elif self.model_type != 'transformer':
            src_mask = torch.ones((mem.shape[0], 1, self.src_len))

        if 'gpu' in self.params['HARDWARE']:
            src_mask = src_mask.cuda()
            decoded = decoded.cuda()
            tgt = tgt.cuda()

        self.model.eval()
        for i in range(max_len):
            if verbose:
                if i%print_step==0: print("decoding sequences of max length ",max_len,"current position: ",i)
            if self.model_type == 'transformer':
                decode_mask = Variable(subsequent_mask(decoded.size(1)).long())
                if 'gpu' in self.params['HARDWARE']:
                    decode_mask = decode_mask.cuda()
                out = self.model.decode(mem, src_mask, Variable(decoded),decode_mask)
            else:
                out, _ = self.model.decode(tgt, mem)
            out = self.model.generator(out)
            prob = F.softmax(out[:,i,:], dim=-1)
            _, next_word = torch.max(prob, dim=1)
            next_word += 1  # to ignore predicting a start token
            tgt[:,i+1] = next_word
            if self.model_type == 'transformer':
                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
            
            if return_probabilities:
                probabilities[:,i,:] = prob
        
        decoded = tgt[:,1:]
        if return_probabilities:
            return decoded, probabilities    
        else:
            return decoded

    def reconstruct(self, data, method='greedy', log=True, return_mems=True, return_str=True, return_probabilities=False):
        """
        Method for encoding input smiles into memory and decoding back
        into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
            return_probabilities (bool): If true, returns probabilities of each token at each position
                                        useful for computing perplexity/entropy/etc.
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        with torch.no_grad():
            data = vae_data_gen(data,
                                max_len=self.src_len,
                                name=self.name, 
                                char_dict=self.params['CHAR_DICT'],
                                d_pp_out=self.params['d_pp_out'])
            data_iter = torch.utils.data.DataLoader(data,
                                                    batch_size=self.params['BATCH_SIZE'],
                                                    shuffle=False, num_workers=0,
                                                    pin_memory=False, drop_last=True)
            self.batch_size = self.params['BATCH_SIZE']
            self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']

            self.model.eval()
            decoded_sequences = []
            decoded_properties = torch.empty((data.shape[0],self.params["d_pp_out"]))
            mems = torch.empty((data.shape[0], self.params['d_latent']))
            for j, data in enumerate(data_iter):
                if log:
                    log_file = open('calcs/{}_progress.txt'.format(self.name), 'a')
                    log_file.write('{}\n'.format(j))
                    log_file.close()
                for i in range(self.params['BATCH_CHUNKS']):
                    batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data[:,:-self.params["d_pp_out"]]
                    src = Variable(mols_data).long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    if 'gpu' in self.params['HARDWARE']:
                        src = src.cuda()
                        src_mask = src_mask.cuda()
                    print(self.device)
                    ### Run through encoder to get memory
                    if self.model_type == 'transformer':
                        _, mem, _, _ = self.model.encode(src, src_mask)
                    elif self.model_type == 'aae': #For both aae and wae the latent memory is not "mu" as it is in vae's case
                        mem, _, _ = self.model.encode(src)
                    elif self.model_type == 'wae':
                        mem, _, _ = self.model.encode(src)
                    else:
                        _, mem, _ = self.model.encode(src)                           
                    if self.params['property_predictor']:
                        props = self.model.predict_property(mem, torch.tensor(0.))
                    else:
                        props=torch.tensor(0)
                    
                    ### grab the batch outputs and store them   
                    start = j*self.batch_size+i*self.chunk_size
                    stop  = j*self.batch_size+(i+1)*self.chunk_size
                    decoded_properties[start:stop] = props
                    mems[start:stop, :] = mem.detach().cpu()
                    
                    ### Decode logic
                    if method == 'greedy':
                        decoded = self.greedy_decode(
                            mem, 
                            src_mask=src_mask, 
                            return_probabilities=return_probabilities
                        )
                        if return_probabilities:
                            decoded, probabilities = decoded # unpack tuple
                    else:
                        decoded = None

                    if return_str:
                        decoded = decode_mols(decoded, self.params['ORG_DICT'])
                        decoded_sequences += decoded
                    else:
                        decoded_sequences.append(decoded)

            if return_mems and not return_probabilities:
                return decoded_sequences, decoded_properties, mems.detach().numpy()
            elif not return_mems and return_probabilities:
                return decoded_sequences, decoded_properties, probabilities
            elif return_mems and return_probabilities:
                return decoded_sequences, decoded_properties, mems.detach().numpy(), probabilities
            else:
                return decoded_sequences, decoded_properties

    def sample(self, n, method='greedy', sample_mode='rand',
                        sample_dims=None, k=None, return_str=True):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k=k)

        if 'gpu' in self.params['HARDWARE']:
            mem = mem.cuda()

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem)
        else:
            decoded = None

        if return_str:
            decoded = decode_mols(decoded, self.params['ORG_DICT'])
        return decoded

    def calc_mems(self, data, log=True, save_dir='memory', save_fn='model_name', save=True):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        data = vae_data_gen(data, 
                            max_len=self.src_len,
                            name=self.name,props=None, 
                            char_dict=self.params['CHAR_DICT'],
                            d_pp_out=self.params["d_pp_out"])

        data_iter = torch.utils.data.DataLoader(data,
                                                batch_size=self.params['BATCH_SIZE'],
                                                shuffle=False, num_workers=0,
                                                pin_memory=False, drop_last=True)
        save_shape = len(data_iter)*self.params['BATCH_SIZE']
        self.batch_size = self.params['BATCH_SIZE']
        self.chunk_size = self.batch_size // self.params['BATCH_CHUNKS']
        mems = torch.empty((save_shape, self.params['d_latent'])).cpu()
        mus  = torch.empty((save_shape, self.params['d_latent'])).cpu()
        logvars = torch.empty((save_shape, self.params['d_latent'])).cpu()
        self.model.eval()

        for j, data in enumerate(data_iter):
            if log:
                log_file = open('memory/{}_progress.txt'.format(self.name), 'a')
                log_file.write('{}\n'.format(j))
                log_file.close()
            for i in range(self.params['BATCH_CHUNKS']):
                batch_data = data[i*self.chunk_size:(i+1)*self.chunk_size,:]
                mols_data  = batch_data[:,                        :-self.params["d_pp_out"]]
                props_data = batch_data[:,-self.params["d_pp_out"]:                        ]
                if 'gpu' in self.params['HARDWARE']:
                    mols_data = mols_data.cuda()
                    props_data = props_data.cuda()

                src = Variable(mols_data).long()
                src_mask = (src != self.pad_idx).unsqueeze(-2)
                ### Run through encoder to get memory
                if self.model_type == 'transformer':
                    with torch.no_grad():
                        mem, mu, logvar, _ = self.model.encode(src, src_mask)
                else:
                    with torch.no_grad():
                        mem, mu, logvar = self.model.encode(src)
                start = j*self.batch_size+i*self.chunk_size
                stop = j*self.batch_size+(i+1)*self.chunk_size
                mems[start:stop, :] = mem.detach().cpu()
                mus[start:stop, :] = mu.detach().cpu()
                logvars[start:stop, :] = logvar.detach().cpu()

        if save:
            if save_fn == 'model_name':
                save_fn = self.name
            save_path = os.path.join(save_dir, save_fn)
            np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
            np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
            np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
        else:
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()

############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    NEED TO MAKE THIS GENERALIZEABLE IT IS HARD SET TO 64*9 = 576 from an input vector of length 128 
    """
    def __init__(self, size, src_len):
        super().__init__()
        conv_layers = []
        self.conv_list = [] # this will allow a flexible model input by changing the decoder shape to match each level of convolution
        in_d = size
        first = True
        input_shape = src_len
        self.out_channels = 64
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9 #OG_9
                first = False
            else:
                kernel_size = 8 #OG_8
            if i == 2:
                out_d = self.out_channels
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(kernel_size=2)))
            in_d = out_d
            #conv_out_shape [(W−K+2P)/S]+1 ;W:input, K:kernel_size, P:padding, S:stride default=1
            #maxpool output shape [(W+2p-D*(K-1)-1)/S]+1  W:input, D:dilation, K:kernel_size, P:padding, S:stride default=kernel_size
            conv_out_shape = ((input_shape-kernel_size)//1)+1 
            maxpool_out_shape = ((conv_out_shape-(2-1)-1)//2)+1
            input_shape = maxpool_out_shape
            self.conv_list.append(input_shape)#save the output shape
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x))
        return x

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    def __init__(self, size, src_len, conv_list):
        super().__init__()
        deconv_layers = []
        in_d = 64
        input_shape = src_len+1
        conv_list.insert(0,input_shape) #add the original source length to the conv shape list
        for i in range(3):
            #formula to find appropriate kernel size for each layer:(L_out-1)-2(L_in-1)+1=K ,K:kernel_size,L_out:new_shape,L_in:old_shape
            L_in = conv_list[3-i]
            L_out = conv_list[3-(i+1)]
            out_d = (size - in_d) // 4 + in_d
            stride = 2
            kernel_size = (L_out-1)-2*(L_in-1)+1
            if i == 2:
                out_d = size
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                  stride=stride, padding=0)))
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.leaky_relu(deconv(x))
        return x

############## Generator #################
class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)

############## Property Predictor #################

class PropertyPredictor(nn.Module):
    "Optional property predictor module. Choice between: decision_tree and deep_net"
    def __init__(self, d_pp, depth_pp, d_latent, type_pp, d_pp_out=1, prediction_types=None):
        super().__init__()
        self.type_pp=type_pp
        self.d_pp_out = d_pp_out
        self.prediction_types = prediction_types

        if "decision_tree" in self.type_pp:
            from sklearn.tree import DecisionTreeClassifier
            self.decision_tree = DecisionTreeClassifier(max_depth=depth_pp)
     
        else:
            prediction_layers = []
            for i in range(depth_pp):
                if i == 0:
                    linear_layer = nn.Linear(d_latent, d_pp)
                elif i == depth_pp - 1:
                    linear_layer = nn.Linear(d_pp, d_pp_out)
                else:
                    linear_layer = nn.Linear(d_pp, d_pp)
                prediction_layers.append(linear_layer)
            self.prediction_layers = ListModule(*prediction_layers)

    def _last_layer_nn(self, x, prediction_layer):
        condn1 = (self.prediction_types is None)
        condn_all_class = ( set(self.prediction_types)==set(["classification"]) )
        condn_all_regre = ( set(self.prediction_types)==set(["regression"]) )
        condn2 = (self.d_pp_out==1)
        if (condn1 or condn_all_class) and condn2: # one classification output
            return torch.sigmoid(prediction_layer(x))
        elif (condn1 or condn_all_class) and not condn2: # multiple classification outputs
            return torch.softmax(prediction_layer(x), dim=-1)
        elif condn_all_regre: # full regression output
            return prediction_layer(x)
        else: # full regression output
            return prediction_layer(x)

    def forward(self, x, true_prop):
        if "decision_tree" in self.type_pp:
            print(x.shape, true_prop.shape)
            self.decision_tree.fit(x.detach().numpy(), true_prop)
            print("N_classes:",self.decision_tree.n_classes_)
            x = self.decision_tree.predict_proba(x.detach().numpy()) #score on training data
        else: #using neural network
            condn3 = (self.prediction_types is None)
            for idx, prediction_layer in enumerate(self.prediction_layers):
                condn1 = (idx == len(self.prediction_layers)-1)
                if condn1:
                    x = self._last_layer_nn(x, prediction_layer)
                else:
                    x = F.leaky_relu(prediction_layer(x))
        return x

############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings. Importantly this embedding is learnable! Weights change with backprop."
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model) #Square root is for the transformer model to keep num's low

class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (pytorch)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        return self.bn(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))
