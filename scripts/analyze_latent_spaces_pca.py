import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pickle as pkl
import seaborn as sns
import time
import torch
import torch.nn as nn

import coranking #coranking.readthedocs.io

from sklearn.decomposition import PCA
from sklearn import metrics
from coranking.metrics import trustworthiness, continuity, LCMC
from transvae.snc import SNC 

from scipy.stats import pearsonr

from transvae import trans_models
from transvae.transformer_models import TransVAE
from transvae.tvae_util import *

def make_model_name(prop, model_number, semi_sup_percent):
    if (semi_sup_percent==100): # don't think this one is correct
        percent = ""
        suffix="dPP64-ZScore"
    elif (semi_sup_percent=='0'):
        if prop in ["log10mic", 'predicted-log10mic']:
            percent="0-"
        else:
            percent = ""
        
        suffix  = "cdhit90-zScoreNormalized"
    else:
        percent = str(semi_sup_percent)+"-"
        suffix="cdhit90-zScoreNormalized"

    model_name=f"transvae-64-peptides-{prop}-zScoreNormalized-{percent}organized-{suffix}"

    return model_name

def encode_seqs(df, stoi):
    encoded_seqs = []
    max_len = 101
    for seq in df['peptides']:
        temp_ = [stoi[aa] for aa in seq]
        seq_len = len(seq)
        temp_ += [stoi["<end>"]]
        temp_ += [stoi["_"]] * (max_len - seq_len)
        temp_ = [0] + temp_
        encoded_seqs.append(temp_)
    df = pd.DataFrame({"encoded_peptides":encoded_seqs})
    return df

def decode_seq(encoded_seq, stoi):
    itos = {v:k for k,v in stoi.items()}
    decoded_seq = []
    for tok in encoded_seq:
        decoded_seq.append(itos[tok])
    decoded_seq = "".join(decoded_seq)
    decoded_seq = decoded_seq.strip("_")
    decoded_seq = decoded_seq.strip("<start>")
    decoded_seq = decoded_seq.strip("<end>")
    return decoded_seq

def run_inference_batch(model, data, batch_size=4000, encode_only=True):
    """
    Run inference on data in batches

    Parameters
    ----------
    model : pytorch object
        model object
    data : numpy array, first column is the sequences, the rest are properties
        data to run inference on
    batch_size : int, optional
        amount to run in each batch, by default 2000
    encode_only : bool, optional
        whether to only encode the data, by default True. If False, then also will decode from the latent space
    
    Returns
    -------
    z : numpy array
        latent space representation
    mu : numpy array
        mean of the latent space
    logvar : numpy array
        log variance of the latent space
    decoded_seqs : numpy array
        decoded sequences. None if encode_only=True
    predicted_props : numpy array
        predicted properties. None if encode_only=True
    """
    n_batches = len(data)/batch_size
    n_batches = int( math.ceil(n_batches) )

    sequences = data.peptides.values.reshape(-1,1)
    fctn      = data.amp.values.reshape(-1,1)
    props     = data.iloc[:,2:].values.reshape(-1,3)

    z_list = []
    mu_list = []
    logvar_list = []
    decoded_seqs_list = []
    predicted_props_list = []
    t0 = time.time()
    t00 = time.time()
    for i in range(n_batches):
        print(f"inference on batch {i+1}/{n_batches}")
        # batch       = data.iloc[i*batch_size:(i+1)*batch_size, 0]
        # batch_props = data.iloc[i*batch_size:(i+1)*batch_size, 1:]
        batch = sequences[  i*batch_size:(i+1)*batch_size]
        batch_props = props[i*batch_size:(i+1)*batch_size]
        
        #if torch.cuda.is_available():
        #    batch = torch.tensor(batch).to(model.device)
        #    batch_props = torch.tensor(batch_props).to(model.device)

        model.params["BATCH_SIZE"] = len(batch)

        with torch.no_grad():
            z, mu, logvar = model.calc_mems(batch, log=False, save=False)

            if not encode_only:
                print("reconstructing..")
                decoded_seqs, predicted_props  = model.reconstruct(np.c_[batch, batch_props],log=False,return_mems=False)
                decoded_seqs_list.append(decoded_seqs)
                predicted_props_list.append(predicted_props)
            else:
                predicted_props = model.model.predict_property(torch.from_numpy(mu), torch.tensor(0.))
                predicted_props_list.append(predicted_props)

            tf = time.time()
            print(f"time per batch = {round(tf-t00,5)}s")

            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)
            t00 = tf

    print(f"time elapsed = {round(time.time()-t0,5)}s")

    if encode_only:
        decoded_seqs = None
    
    z      = np.concatenate(z_list, axis=0)
    mu     = np.concatenate(mu_list, axis=0)
    logvar = np.concatenate(logvar_list, axis=0)
    predicted_props = np.concatenate(predicted_props_list, axis=0)
    if not encode_only:
        decoded_seqs = np.concatenate(decoded_seqs_list, axis=0)

    return z, mu, logvar, decoded_seqs, predicted_props


def main_latent_space_analysis(model_number, semi_sup_percent, prop, data_args, checkpointz_dir="../ckpt_local_pkg/"):

    _use_full_test_data = True
    _use_full_train_data= True
    ########################################################
    # make model name
    ########################################################
    _prop=prop # options: ['boman', chargepH7p2, hydrophobicity, boman-chargepH7p2, bch]
    output_dir = data_args["output_dir"]

    model_name = make_model_name(_prop, model_number, semi_sup_percent)
    print(f"{model_name=}")

    model_src = checkpointz_dir + model_name + f"/{model_number}_{model_name}.ckpt"

    ########################################################
    # Load data 
    ########################################################

    char_dict_fpath    = data_args["char_dict_fpath"]
    char_weights_fpath = data_args["char_weights_fpath"]

    train_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_train_analysis_subset.txt")

    if _use_full_test_data:
        print("using full test dataset")
        test_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_test_full.csv")
        # test_data_subset = torch.load(f"{output_dir}peptides_2024_cdhit90_test_full.pt")
    else:
        test_data_subset  = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_test_analysis_subset.txt")

    if _use_full_train_data:
        print("using full train dataset")
        train_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_train_full.csv")
        # test_data_subset = torch.load(f"{output_dir}peptides_2024_cdhit90_test_full.pt")
    else:
        train_data_subset  = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_train_analysis_subset.txt")


    #########################################
    # load a trained generative model
    print("loading generative model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cpu")
    device_str = "cpu"

    model_obj=torch.load(model_src, map_location=device)
    model = TransVAE(load_fn=model_src, workaround=device_str, params=model_obj['params'])
    # model = RNN(load_fn=model_src, workaround="cpu")
    model.params['HARDWARE']= device_str

    model.model.eval()
    batch_size = 4000
    _do_train_set = True
    _do_test_set  = True
    if _do_train_set:
        print("running inference on train data")
        z, mu, logvar, decoded_seqs, predicted_props = run_inference_batch(model, train_data_subset, batch_size=batch_size, encode_only=True)

        ###
        # saving
        print("saving..")
        os.makedirs(output_dir+"/"+model_name, exist_ok=True)
        np.save(f"{output_dir}/{model_name}/z_{model_name}_train.npy", z)
        np.save(f"{output_dir}/{model_name}/mu_{model_name}_train.npy", mu)
        np.save(f"{output_dir}/{model_name}/logvar_{model_name}_train.npy", logvar)
        np.save(f"{output_dir}/{model_name}/predicted_props_{model_name}_train.npy", predicted_props)

    if _do_test_set:
        print("running inference on test data")
        z, mu, logvar, decoded_seqs, predicted_props = run_inference_batch(model, test_data_subset, batch_size=batch_size, encode_only=True)
    
        ###
        # saving
        print("saving test data outputs..")
        os.makedirs(output_dir+"/"+model_name, exist_ok=True)
        np.save(f"{output_dir}/{model_name}/z_{model_name}_test.npy", z)
        np.save(f"{output_dir}/{model_name}/mu_{model_name}_test.npy", mu)
        np.save(f"{output_dir}/{model_name}/logvar_{model_name}_test.npy", logvar)
        np.save(f"{output_dir}/{model_name}/predicted_props_{model_name}_test.npy", predicted_props)
    else:
        print("not running inference on test set")

    if decoded_seqs is not None:
        np.save(f"{output_dir}/{model_name}/decoded_seqs_{model_name}_train.npy", decoded_seqs)

def correlation_analysis(pca_mu, prop_vec, verbose=False):
    """
    Calculate the correlation between the PCA'd latent space and the property vector

    Parameters
    ----------
    pca_mu : numpy array
        PCA of latent space
    prop_vec : numpy array
        property vector

    Returns
    -------
    coef1 : float 
        max-1 correlation coefficient
    coef2 : float
        max-2 correlation coefficient (the second highest)
    idx1 : int
        PC index of max-1
    idx2 : int
        PC index of max-2
    """
    c=prop_vec
    mask = ~np.isnan(c)
    #mask = (c.flatten()!=np.nan)
    print(f"{len(c)=}, {len(pca_mu)=}")
    statistic2pc = {}
    for i in range(5):
        _pc = pca_mu[:,i]
        _pr = pearsonr(_pc[mask], c.flatten()[mask])
        
        statistic2pc[ abs(_pr.statistic)  ] = i
        
        if verbose:
            print(f"PC{i+1} has pearson r={_pr}")

    # sort statistics
    _sorted = sorted( list(statistic2pc.keys()) )
    max1=_sorted[-1]
    max2=_sorted[-2]

    max1pc=statistic2pc[max1]
    max2pc=statistic2pc[max2]
    
    if verbose:
        print("check lin combinations of two PCs")
    
        for i in range(5):
            for j in range(i,5):
                _pc = pca_mu[:,i] + pca_mu[:,j]
                _pr = pearsonr(_pc, c.flatten())
                if verbose:
                    print(f"PC{i+1}+PC{j+1} has pearson r={_pr}")
    
    return max1, max2, max1pc, max2pc

def plot_pca_latent_space(pca_mu, pca_idx1, pca_idx2, c, _prop,data_args, suffix=""):
    marker_size = 24
    fontsize=16
    marker_sizes = [marker_size for _ in range(len(pca_mu[:,0]))]
    # _prop=data_args['prop']
    semi_sup_percent = data_args['semi_sup_percent']
    SAVE_FIGURES = data_args['save_figures']

    _property=_prop # choices = ["amp","boman", "charge", "mic", "hydrophobicity", seq_length]
    if _property=="amp":
        _property = "AMP/non-AMP"
    elif _property=="boman":
        _property = "Boman Index"
    elif _property=="mic":
        _property = "log10(mic)"
        # c = ecoli_avg_mic['log10mic'].values
    elif _property=="hc50":
        _property = "log10(hc50)"
        # c = ecoli_avg_mic['log10hc50'].values
    elif _property=="charge" or _property=="chargepH7p2":
        _property = "Charge (pH=7.2)"
    elif _property == "hydrophobicity":
        pass
    elif _property == "seq_length":
        _property = "Sequence Length"

    plt.figure(figsize=(12,10))
    _palette_name = "magma"
    cmap = sns.color_palette("magma", as_cmap=True) # sns.color_palette("dark:salmon_r", as_cmap=True)
    cmap = sns.color_palette("dark:salmon_r", as_cmap=True)
    ax = sns.scatterplot(
                x=pca_mu[:,pca_idx1], 
                y=pca_mu[:,pca_idx2], 
                hue=c.flatten() ,
                s=marker_size,#marker_sizes,
                palette=_palette_name,
    )
    norm=plt.Normalize(c.flatten().min(), c.flatten().max())
    sm = plt.cm.ScalarMappable(cmap=_palette_name, norm=norm)
    sm.set_array([])

    sns.kdeplot(
        x=pca_mu[:,pca_idx1],
        y=pca_mu[:,pca_idx2],
        ax=ax,
        levels=5
    )

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm,ax=ax, label=_property)

    plt.xlabel(f"PC {pca_idx1+1}",fontsize=fontsize)
    plt.ylabel(f"PC {pca_idx2+1}",fontsize=fontsize)
    plt.title(f"PCA of latent space, coloured by {_property}",fontsize=fontsize)
    # cbar = plt.colorbar(cmap)
    # cbar = sns.color
    # cbar.set_label(_property, rotation=270, fontsize=fontsize)
    # plt.xlim([-5,5])

    _fname_property = _property.split(" ")[0]
    print(f"{_fname_property=}")
    if SAVE_FIGURES:
        plt.savefig(f"{data_args['output_dir']}/{data_args['model_name']}/pca_projection_transvae-64-{_prop}-{semi_sup_percent}-coloredBy{_fname_property}{suffix}.png",dpi=300)

def plot_predicted_vs_true(predicted_props, predicted_props_test, true_props, true_props_test,_property, data_args, skip_test=False):
    # train_predicted_props, test_predicted_props, prop_vecs[:,_prop_idx], prop_vecs_test[:,_prop_idx], data_args
    fig = plt.figure(figsize=(16,9))
    axes = fig.subplots(1,2)

    # _property=data_args['prop']
    _idx = 0
    if _property in ["boman","Boman Index"]:
        _idx=min(0, predicted_props.shape[1]-1) 
    elif _property in ["Charge (pH=7.2)","charge", "chargepH7p2"]:
        _idx=min(1, predicted_props.shape[1]-1)
    elif _property=="hydrophobicity":
        _idx=min(2, predicted_props.shape[1]-1)
    _pred_prop = predicted_props[:,_idx]
    
    for i in range(2):
        if i==0:
            train_or_test="train"
        else:
            if skip_test:
                continue
            train_or_test="test"
            _pred_prop = predicted_props_test[:,_idx]
            true_props = true_props_test


        axes[i].scatter(true_props, _pred_prop)

        _min_y = min(_pred_prop.flatten() )
        _max_y = max(_pred_prop.flatten() )

        _min_x = min(true_props.astype(float).flatten())
        _max_x = max(true_props.astype(float).flatten())

        _pr = pearsonr(_pred_prop.flatten(), true_props.astype(float).flatten())

        axes[i].set_title(f"Predicted vs True ({train_or_test} set)")
        axes[i].set_xlabel(f"True {_property}")
        axes[i].set_ylabel(f"Predicted {_property}")
        axes[i].text(
            _min_x, 
            _min_y + 0.75*(_max_y-_min_y), 
            f"Pearson CorrCoef={round(_pr.statistic,3)}"
        )

    if data_args['save_figures']:
        fig.savefig(f"{data_args['output_dir']}/{data_args['model_name']}/predicted_vs_true_{_property}.png",dpi=200)

def pca_latent_space_analysis(model_number, data_args):
    use_full_test_set =True
    use_full_train_set=True
    ########################################################
    # make model name
    ########################################################


    output_dir = data_args["output_dir"]
    model_name = data_args["model_name"]

    pca = PCA(n_components=5)
    pca_test = PCA(n_components=5)

    # z_train = np.load(f"{output_dir}/{model_name}/z_{model_name}_train.npy")
    if use_full_train_set:
        mu_train = np.load(f"{output_dir}/{model_name}/mu_{model_name}_train.npy")
    else:
        mu_train = np.load(f"{output_dir}/{model_name}/mu_{model_name}_train.npy")


    if use_full_test_set:
        mu_test  = np.load(f"{output_dir}/{model_name}/mu_{model_name}_test.npy")
    else:
        raise NotImplementedError("using test data subset not implemented yet")
        # mu_test  = np.load(f"{output_dir}/{model_name}/mu_{model_name}_test.npy")
        pass

    # fit PCA
    pca.fit(mu_train)
    pca_mu_train = pca.transform(mu_train)

    with open(f"{output_dir}/{model_name}/pca_explained_var_and_sing_vals_train.txt",'a') as f:
        _header = ",".join([f"explain_var_ratio_comp{i}" for i in range(5)] + [f"singular_value{i}" for i in range(5)])
        _header += "\n"
        _content = ",".join([str(x) for x in list(pca.explained_variance_ratio_) + list(pca.singular_values_) ])
        f.write(_header)
        f.write(_content)

   
    pca_test.fit(mu_test)
    pca_mu_test = pca_test.transform(mu_test)
    with open(f"{output_dir}/{model_name}/pca_explained_var_and_sing_vals_test.txt",'a') as f:
        _header = ",".join([f"explain_var_ratio_comp{i}" for i in range(5)] + [f"singular_value{i}" for i in range(5)])
        _header += "\n"
        _content = ",".join([str(x) for x in list(pca_test.explained_variance_ratio_) + list(pca_test.singular_values_) ])
        f.write(_header)
        f.write(_content)

    # load the property vectors
    if use_full_train_set:
        train_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_train_full.csv")
    else:
        train_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_train_analysis_subset.txt")

    if use_full_test_set:
        test_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_test_full.csv")
    else:
        test_data_subset = pd.read_csv(f"{output_dir}peptides_2024_cdhit90_unbalanced_test_analysis_subset.txt")

    train_predicted_props = np.load(f"{output_dir}/{model_name}/predicted_props_{model_name}_train.npy")
    test_predicted_props  = np.load(f"{output_dir}/{model_name}/predicted_props_{model_name}_test.npy")

    prop_vecs      = train_data_subset.iloc[:,2:].values
    prop_vecs_test = test_data_subset.iloc[ :,2:].values
    
    with open(f"{output_dir}/{model_name}/pca_correlations_{model_name}_train_set.txt", "w") as f:
        f.write("property,max1,max2,max1pc,max2pc\n")
    with open(f"{output_dir}/{model_name}/pca_correlations_{model_name}_test_set.txt", "w") as f:
        f.write("property,max1,max2,max1pc,max2pc\n")

    _prop=data_args['prop'] # options: ['boman', chargepH7p2, hydrophobicity, boman-chargepH7p2, bch]
    if ("-" in _prop) and (_prop != "predicted-log10mic"):
        _props = _prop.split("-")
    elif _prop=="bch":
        _props = ["boman","chargepH7p2", "hydrophobicity"]
    else:
        _props = [_prop]

    for _prop in _props:
        skip_test = False
        if _prop=="boman":
            _prop_idx = 0
        elif _prop=="chargepH7p2" or _prop=="charge":
            _prop_idx = 1
        elif _prop=="hydrophobicity":
            _prop_idx = 2
        elif _prop=="log10mic":
            prop_vecs       = pd.read_csv(f"data/peptides_w_mic_train.txt")
            prop_vecs_test  = pd.read_csv(f"data/peptides_w_mic_test.txt")

            prop_vecs = prop_vecs.log10mic.values.reshape(-1,1)
            prop_vecs[prop_vecs==1000.0] = np.nan

            prop_vecs_test = prop_vecs_test.log10mic.values.reshape(-1,1)
            prop_vecs_test[prop_vecs_test==1000.0] = np.nan
            _prop_idx = 0 
            skip_test=True
        elif _prop=="predicted-log10mic":
            prop_vecs      = pd.read_csv("data/peptides_predicted-log10mic_zScoreNormalized_train.txt")
            prop_vecs_test = pd.read_csv("data/peptides_predicted-log10mic_zScoreNormalized_test.txt")

            prop_vecs = prop_vecs.predicted_mic.values.reshape(-1,1)
            prop_vecs_test = prop_vecs_test.predicted_mic.values.reshape(-1,1)

            _prop_idx = 0
        else:
            raise ValueError(f"invalid property value given. {_prop=}")
        
        ## train set
        # calculate correlations
        max1, max2, max1pc, max2pc = correlation_analysis(pca_mu_train, prop_vecs[:,_prop_idx])
        with open(f"{output_dir}/{model_name}/pca_correlations_{model_name}_train_set.txt", "a") as f:
            f.write(f"{_prop=},{max1=},{max2=},{max1pc=},{max2pc=}\n")

        # make scatter plot of top two PCs
        plot_pca_latent_space(pca_mu_train, max1pc, max2pc, prop_vecs[:,_prop_idx], _prop, data_args, suffix="_train")

        ## test set
        # calculate correlations
        if not skip_test:
            max1, max2, max1pc, max2pc = correlation_analysis(pca_mu_test, prop_vecs_test[:,_prop_idx], verbose=True)
            with open(f"{output_dir}/{model_name}/pca_correlations_{model_name}_test_set.txt", "a") as f:
                f.write(f"{_prop=},{max1=},{max2=},{max1pc=},{max2pc=}\n")
        
            # make scatter plot of top two PCs
            plot_pca_latent_space(pca_mu_test, max1pc, max2pc, prop_vecs_test[:,_prop_idx], _prop, data_args, suffix="_test")


        ## plot predicted vs true of properties
        plot_predicted_vs_true(train_predicted_props, test_predicted_props, prop_vecs[:,_prop_idx], prop_vecs_test[:,_prop_idx], _prop, data_args, skip_test=skip_test)


def make_learning_curves():
    df = pd.read_csv(f"{trials_dir}/log_{model_name}.txt", sep=",")
    print(df.shape)
    print("max N epochs =", df.epoch.max() )

    cols2grab = ["epoch","tot_loss", "recon_loss","kld_loss","prop_bce_loss"]

    grouped_by_epoch_train = df.loc[df["data_type"]=="train"][cols2grab].groupby('epoch').mean()
    grouped_by_epoch_test  = df.loc[df["data_type"]== "test"][cols2grab].groupby('epoch').mean()

    # sort of baseline cross-entropy loss if guess characters uniformly
    alphabet_size = 20 # number of classes, in peptide case: 20 amino acids
    baseline_loss = -np.log(1/alphabet_size) 

    beta_init = 1e-8
    beta_final= 0.05
    beta_anneal_m = (beta_final-beta_init)/(200)

    for col_ in cols2grab:
        if col_=="epoch":
            continue
        plt.plot(grouped_by_epoch_train[col_], label=col_)
    plt.hlines(
        y=baseline_loss, 
        xmin=df['epoch'].min(), 
        xmax=df['epoch'].max(),
        label="ReconBaseline",
        color="k",
        linestyle="--"
    )

    plt.grid(axis="y")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("Training losses (organized by log10mic)")
    plt.legend()
    plt.ylim([-0.05,5])
    if SAVE_FIGURES:
        plt.savefig("figures/transvae-64-isometry-train.png",dpi=200)

    ########################################################
    # Validation set
    ########################################################
    for col_ in cols2grab:
        if col_=="epoch":
            continue
        plt.plot(grouped_by_epoch_test[col_], label=col_)
    plt.hlines(
        y=baseline_loss, 
        xmin=df['epoch'].min(), 
        xmax=df['epoch'].max(),
        label="ReconBaseline",
        color="k",
        linestyle="--"
    )
    plt.grid(axis="y")
    plt.title("Validation set loss (organized by log10mic)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend()
    plt.ylim([-0.1,5])
    if SAVE_FIGURES:
        plt.savefig("figures/transvae-64-isometry-validation.png",dpi=200)


def make_data_subset(data_args, random_seed=42):
    
    char_dict_fpath    = data_args["char_dict_fpath"]
    char_weights_fpath = data_args["char_weights_fpath"]
    train_seqs_fpath   = data_args["train_seqs"]
    train_fctn_fpath   = data_args["train_fctn"]
    train_prop_fpath   = data_args["train_prop"]

    test_seqs_fpath   = data_args["test_seqs"]
    test_fctn_fpath   = data_args["test_fctn"]
    test_prop_fpath   = data_args["test_prop"]
    
    balance_amps = data_args["balance_amps"]
    n_total      = data_args["n_total"]

    # char_dict_fpath   =data_full_dir+"vocab/char_dict_peptides_2024_cdhit90_unbalanced.pkl"
    # char_weights_fpath=data_full_dir+"vocab/char_weights_peptides_2024_cdhit90_unbalanced.pkl"
    
    #########################################
    # load all of the files
    with open(char_dict_fpath, 'rb') as f:
        char_dict = pkl.load(f)

    char_weights = np.load(char_weights_fpath)

    train_seqs = pd.read_csv(train_seqs_fpath)
    train_fctn = pd.read_csv(train_fctn_fpath)
    train_prop = pd.read_csv(train_prop_fpath)

    test_seqs = pd.read_csv(test_seqs_fpath)
    test_fctn = pd.read_csv(test_fctn_fpath)
    test_prop = pd.read_csv(test_prop_fpath)

    n_props=3
    print(f"{train_prop.shape=}")
    print(f"{train_seqs.shape=}")

    print(f"{test_prop.shape=}")
    print(f"{test_seqs.shape=}")

    df_train = encode_seqs(train_seqs, char_dict)    
    df_train["amp_or_not"] = train_fctn['amp']
    
    df_train[         "boman"] = train_prop["boman"]
    df_train["charge(pH=7.2)"] = train_prop["charge(pH=7.2)"]
    df_train['hydrophobicity'] = train_prop['hydrophobicity']

    # make test dataframes
    df_test = encode_seqs(test_seqs, char_dict)
    df_test["amp_or_not"] = test_fctn['amp']

    df_test[         "boman"] = test_prop["boman"]
    df_test["charge(pH=7.2)"] = test_prop["charge(pH=7.2)"]
    df_test['hydrophobicity'] = test_prop['hydrophobicity']
    
    for idx,df in enumerate([df_train, df_test]):
        if idx==0:
            print("train")
        else:
            print("test")
        sampled_peptides = []
        sampled_fctns = []
        sampled_props = []
        if balance_amps:
            n_each = n_total//2
            n_amps     = n_each
            n_non_amps = n_each
            some_amps     = df[df.amp_or_not==1].sample(    n_amps, random_state=random_seed)
            some_non_amps = df[df.amp_or_not==0].sample(n_non_amps, random_state=random_seed)

            for i,seq in enumerate(some_amps["encoded_peptides"]):
                sampled_peptides.append( [decode_seq(seq,char_dict)] )
                sampled_fctns.append( [some_amps.iloc[i,1]])
                sampled_props.append( [some_amps.iloc[i,2:] ]) 

            for i,seq in enumerate(some_non_amps["encoded_peptides"]):
                sampled_peptides.append( [decode_seq(seq,char_dict)] )
                sampled_fctns.append( [some_non_amps.iloc[i,1]])
                sampled_props.append( [some_non_amps.iloc[i,2:] ])
        else:
            some_peptides = df.sample(n_total, random_state=random_seed)
            for i,seq in enumerate(some_peptides["encoded_peptides"]):
                sampled_peptides.append( [decode_seq(seq,char_dict)] )
                sampled_fctns.append( [some_peptides.iloc[i,1]])
                sampled_props.append( [some_peptides.iloc[i,2:] ])

        sampled_peptides = np.array(sampled_peptides)
        sampled_fctns    = np.array(sampled_fctns)
        sampled_props    = np.array(sampled_props).reshape(-1,n_props)

        df_subset = pd.DataFrame(np.c_[sampled_peptides, sampled_fctns, sampled_props], columns=["peptides","amp","boman","charge(pH=7.2)","hydrophobicity"])

        print("saving..")
        if idx==0:
            print("train subset")
            df_subset.to_csv(f"{data_args['output_dir']}peptides_2024_cdhit90_unbalanced_train_analysis_subset.txt", index=False)
        else:
            print("test_subset")
            df_subset.to_csv(f"{data_args['output_dir']}peptides_2024_cdhit90_unbalanced_test_analysis_subset.txt", index=False)

def compute_distortion_metrics(mus, pca_batch):

    save_df = pd.DataFrame()
    
    trust_subsamples = []
    cont_subsamples = []
    lcmc_subsamples = []
    steadiness_subsamples = []
    cohesiveness_subsamples = []
    
    t0=time.time()
    n=35
    expected_time = n*9
    time_elapsed  = 0
    parameter = { "k": 50,"alpha": 0.1 } #for steadiness and cohesiveness
    for s in range(n):
        s_len = len(mus)//n #sample lengths
        Q = coranking.coranking_matrix(mus[s_len*s:s_len*(s+1)], pca_batch[s_len*s:s_len*(s+1)])
        trust_subsamples.append( np.mean(trustworthiness(Q, min_k=1, max_k=50)) )
        cont_subsamples.append( np.mean(continuity(Q, min_k=1, max_k=50)) )
        lcmc_subsamples.append( np.mean(LCMC(Q, min_k=1, max_k=50)) )
        # print(n,trust_subsamples[s],cont_subsamples[s],lcmc_subsamples[s])
        
        metrics = SNC(raw=mus[s_len*s:s_len*(s+1)], emb=pca_batch[s_len*s:s_len*(s+1)], iteration=300, dist_parameter=parameter)
        metrics.fit() #solve for steadiness and cohesiveness
        steadiness_subsamples.append(metrics.steadiness())
        cohesiveness_subsamples.append(metrics.cohesiveness())
        # print(metrics.steadiness(),metrics.cohesiveness())
        Q=0 #trying to free RAM
        metrics=0
        torch.cuda.empty_cache() #free allocated CUDA memory
    
        if s%5==0:
            print(f"on {s}/{n}")
            dt = time.time() - t0
            if s==0:
                expected_time = (n-1)*dt
                print(f"expected to take {expected_time}s")
                
            time_elapsed += dt
            print(f"{time_elapsed}s have elapsed | {expected_time- time_elapsed}s likely remaining")
            t0=time.time()
    
    save_df = pd.DataFrame({'latent_to_PCA_trustworthiness':trust_subsamples})
    # save_df = pd.concat([save_df,pd.DataFrame({'latent_to_PCA_trustworthiness':trust_subsamples})], axis=1)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_to_PCA_continuity':cont_subsamples})], axis=1)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_to_PCA_lcmc':lcmc_subsamples})], axis=1)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_to_PCA_steadiness':steadiness_subsamples})], axis=1)
    save_df = pd.concat([save_df,pd.DataFrame({'latent_to_PCA_cohesiveness':cohesiveness_subsamples})], axis=1)
    
    return save_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_step", type=str, default="data_subset")
    parser.add_argument("--model_number", type=int, default="020")
    parser.add_argument("--trials_dir", type=str, default="trials")
    parser.add_argument("--checkpointz_dir", type=str, default="checkpoints")
    parser.add_argument("--save_figures", type=str, default="no")
    
    args = parser.parse_args()

    SAVE_FIGURES = args.save_figures
    if SAVE_FIGURES=="yes":
        SAVE_FIGURES=True
    else:
        SAVE_FIGURES=False

    random_seed = 42

    data_args = {}
    # data_dir = "../data_full/final/"
    data_dir = "data/"
    data_args['output_dir'] = "analysis/"
    data_args["char_dict_fpath"]    = data_dir+"vocab/char_dict_peptides_2024_cdhit90_unbalanced.pkl"
    data_args["char_weights_fpath"] = data_dir+"vocab/char_weights_peptides_2024_cdhit90_unbalanced.npy"

    # train
    data_args["train_seqs"]         = data_dir+"peptides_2024_cdhit90_unbalanced_train.txt"
    data_args["train_fctn"]         = data_dir+"peptides_2024_cdhit90_unbalanced_train_function.txt"
    data_args["train_prop"]         = data_dir+"properties-zScoreNormalized/peptides_2024_cdhit90_unbalanced_train_properties_zScoreNormalized.txt"
    
    # test
    data_args["test_seqs"]          = data_dir+"peptides_2024_cdhit90_unbalanced_test.txt"
    data_args["test_fctn"]          = data_dir+"peptides_2024_cdhit90_unbalanced_test_function.txt"
    data_args["test_prop"]          = data_dir+"properties-zScoreNormalized/peptides_2024_cdhit90_unbalanced_test_properties_zScoreNormalized.txt"
    data_args["balance_amps"]       = True
    data_args["n_total"]            = 10_000
    
    data_args['save_figures'] = SAVE_FIGURES
    
    percents= ['25','50','75','98'] #might be able to include "100" as well
    percents= ['0']
    percents= ['98']
    
    props = ['boman', 'chargepH7p2', 'hydrophobicity', 'boman-chargepH7p2', 'bch', 'log10mic', 'predicted-log10mic']
    props = ['predicted-log10mic']

    if args.pipeline_step=="data_subset":
        make_data_subset(data_args, random_seed=random_seed)
    elif args.pipeline_step=="model_inference":
        model_number = args.model_number
        if len(str(model_number)) < 3:
            model_number = f"0{model_number}"

        for prop in props:
            for semi_sup_percent in percents:
                data_args["semi_sup_percent"] = semi_sup_percent
                data_args["prop"] = prop
                data_args["model_name"] = make_model_name(prop, model_number, semi_sup_percent)
                try:
                    print()
                    print(f"Running inference on {prop} with {semi_sup_percent}% semi-supervised data")
                    main_latent_space_analysis(model_number, semi_sup_percent, prop, data_args, checkpointz_dir=args.checkpointz_dir)
                except Exception as e:
                    # check if keyboard interrupt
                    if "KeyboardInterrupt" in str(e):
                        break
                    print(f"Error: {e}")        
    elif args.pipeline_step=="pca":
        data_args["save_figures"] = True
        model_number = args.model_number
        if len(str(model_number)) < 3:
            model_number = f"0{model_number}"
        
        for prop in props:
            for semi_sup_percent in percents:
                data_args["semi_sup_percent"] = semi_sup_percent
                data_args["prop"] = prop
                data_args["model_name"] = make_model_name(prop, model_number, semi_sup_percent)
                print()
                print(f"Running PCA on {prop} with {semi_sup_percent}% semi-supervised data")
                try:
                    pca_latent_space_analysis(model_number, data_args)
                except FileNotFoundError as e:
                    print(f"Error: {e}")
    elif args.pipeline_step=="manifold_distortion":

        model_number = args.model_number
        if len(str(model_number)) < 3:
            model_number = f"0{model_number}"

        t0 = time.time()
        for prop in props:
            for semi_sup_percent in percents:
                
                model_name = make_model_name(str(prop), model_number, semi_sup_percent)
                mu_fpath = f"analysis/{model_name}/mu_{model_name}_test.npy"
                print(mu_fpath)
        
                try:
                    mus = np.load(mu_fpath)
                except:
                    print(f"failed to load: {model_name=}")
                    print("continuing")
                    continue
                #create random index and re-index ordered memory list creating n random sub-lists (ideally resulting in IID random lists)
                np.random.seed(seed=42)
                random_idx = np.random.permutation(np.arange(stop=mus.shape[0]), )
                max_amount_of_data_to_use = 1200*35 # n=35 below, 1200 per batch should be statistically powerful
                
                mus = mus[random_idx]
                mus = mus[:max_amount_of_data_to_use]
        
                pca = PCA(n_components=5)
                pca_batch =pca.fit_transform(mus)
        
                save_df = compute_distortion_metrics(mus, pca_batch)
                print(f"saving distortion quantities for {model_name=}")
                print(f"time elapsed: {time.time()-t0}s")
                save_df.to_csv(f"analysis/{model_name}/latent_to_pca_distortions.csv",index=False)
                t0 = time.time()            
    elif args.pipeline_step=="learning_curves":
        make_learning_curves()
    print("Done.")
