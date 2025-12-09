import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

from transvae.tvae_util import KLAnnealer
from scipy.stats import pearsonr
# Plotting functions

def plot_test_train_curves(paths, target_path=None, loss_type='tot_loss', data_type='test', labels=None, colors=None):
    """
    Plots the training curves for a set of model log files

    Arguments:
        paths (list, req): List of paths to log files (generated during training)
        target_path (str): Optional path to plot target loss (if you are trying to replicate or improve upon a given loss curve)
        loss_type (str): The type of loss to plot - tot_loss, kld_loss, recon_loss, etc.
        labels (list): List of labels for plot legend
        colors (list): List of colors for each training curve
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']
    if labels is None:
        labels = []
        for path in paths:
            path = path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0]
            labels.append(path)
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        print(df.head(10))
        train_data = df[df.data_type == 'train_loss'].groupby('epoch').mean()[loss_type]
        print(df.head(10))
        test_data = df[df.data_type == 'test_loss'].groupby('epoch').mean()[loss_type]
        print(test_data)
        if loss_type == 'kld_loss':
            klannealer = KLAnnealer(1e-8, 0.05, 2000, 0)
            klanneal = []
            for j in range(2000):
                klanneal.append(klannealer(j))
            train_data /= 1
            test_data /= 1
        plt.plot(train_data, c=colors[i], lw=2.5, label=labels[i], alpha=0.95)
        #plt.plot(test_data, c=colors[i], lw=2.5, label=labels[i], alpha=0.95)
        
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yscale('log')
    #plt.ylabel(loss_type, rotation='horizontal', labelpad=30)
    plt.xlabel('epoch')
    return plt

def plot_loss_by_type(path ,loss_types=['tot_loss', 'recon_loss', 'kld_loss', 'prop_bce_loss'], colors=None):
    """
    Plot the training curve of one model for each loss type

    Arguments:
        path (str, req): Path to log file of trained model
        colors (list): Colors for each loss type
    """
    plt.rc('font', size=14)          # controls default text sizes
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

    if colors is None:
        colors = ['#008080', '#B86953', '#932191', '#90041F', '#0F4935']

    df = pd.read_csv(path)

    plt.figure(figsize=(12,8))
    ax = plt.subplot(111)
    start_pt = 0 #start the graph at a custom index
    end_pt = 2000 # end graph at a custom index
    if None in loss_types:
        loss_types = ['tot_loss', 'recon_loss', 'kld_loss', 'prop_bce_loss']
    for i, loss_type in enumerate(loss_types):
        train_data = df[df.data_type == 'train'].groupby('epoch').mean()[loss_type]
        test_data = df[df.data_type == 'test'].groupby('epoch').mean()[loss_type]
        plt.plot(train_data[start_pt:end_pt], c=colors[i], label='train_'+loss_type)
        plt.plot(test_data[start_pt:end_pt], c=colors[i], label='test_'+loss_type, ls=':')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.title(path.split('/')[-1].split('log_GRUGRU_')[-1].split('.')[0])
    plt.title('Train and Test KLD-Loss RNN-32 beta=0.05')
    return plt

def plot_reconstruction_accuracies(dir, colors=None):
    """
    Plots token, SMILE and positional reconstruction accuracies for all model types in directory

    Arguments:
        dir (str, req): Directory to json files containing stored accuracies for each trained model
        colors (list): List of colors for each trained model
    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)

    smile_accs = {}
    token_accs = {}
    pos_accs = {}
    for k, v in data.items():
        smile_accs[k] = v['accs']['test'][0]
        token_accs[k] = v['accs']['test'][1]
        pos_accs[k] = v['accs']['test'][2]

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12,4), sharey=True,
                                     gridspec_kw={'width_ratios': [1, 1, 2]})
    a0.bar(np.arange(len(smile_accs)), smile_accs.values(), color=colors[:len(smile_accs)])
    a0.set_xticks(np.arange(len(smile_accs)))
    a0.set_xticklabels(labels=smile_accs.keys(), rotation=45)
    a0.set_ylim([0,1])
    a0.set_ylabel('Accuracy', rotation=0, labelpad=30)
    a0.set_title('Per SMILE')
    a1.bar(np.arange(len(token_accs)), token_accs.values(), color=colors[:len(token_accs)])
    a1.set_xticks(np.arange(len(token_accs)))
    a1.set_xticklabels(labels=token_accs.keys(), rotation=45)
    a1.set_ylim([0,1])
    a1.set_title('Per Token')
    for i, set in enumerate(pos_accs.values()):
        a2.plot(set, lw=2, color=colors[i])
    a2.set_xlabel('Token Position')
    a2.set_ylim([0,1])
    a2.set_title('Per Token Sequence Position')
    return fig

def plot_moses_metrics(dir, colors=None):
    """
    Plots tiled barplot depicting the performance of the model on each MOSES metric as a function
    of epoch.

    Arguments:
        dir (str, req): Directory to json files containing calculated MOSES metrics for each model type
        colors (list): List of colors for each trained model

    """
    if colors is None:
        colors = ['#005073', '#B86953', '#932191', '#90041F', '#0F4935']

    data, labels = get_json_data(dir)
    data['paper_vae'] = {'valid': 0.977,
                         'unique@1000': 1.0,
                         'unique@10000': 0.998,
                         'FCD/Test': 0.099,
                         'SNN/Test': 0.626,
                         'Frag/Test': 0.999,
                         'Scaf/Test': 0.939,
                         'FCD/TestSF': 0.567,
                         'SNN/TestSF': 0.578,
                         'Frag/TestSF': 0.998,
                         'Scaf/TestSF': 0.059,
                         'IntDiv': 0.856,
                         'IntDiv2': 0.850,
                         'Filters': 0.997,
                         'logP': 0.121,
                         'SA': 0.219,
                         'QED': 0.017,
                         'weight': 3.63,
                         'Novelty': 0.695,
                         'runtime': 0.0}
    labels.append('paper_vae')
    metrics = list(data['paper_vae'].keys())

    fig, axs = plt.subplots(5, 4, figsize=(20,14))
    for i, ax in enumerate(fig.axes):
        metric = metrics[i]
        metric_data = []
        for label in labels:
            metric_data.append(data[label][metric])
        ax.bar(np.arange(len(metric_data)), metric_data, color=colors[:len(metric_data)])
        ax.set_xticks(np.arange(len(metric_data)))
        ax.set_xticklabels(labels=labels)
        ax.set_title(metric)
    return fig


def get_json_data(dir, fns=None, labels=None):
    """
    Opens and stores json data from a given directory

    Arguments:
        dir (str, req): Directory containing the json files
        labels (list): Labels corresponding to each file
    Returns:
        data (dict): Dictionary containing all data within
                     json files
        labels (list): List of keys corresponding to dictionary entries
    """
    if fns is None:
        fns = []
        for fn in os.listdir(dir):
            if '.json' in fn:
                fns.append(os.path.join(dir, fn))
    if labels is None:
        labels = []
        fn = fn.split('/')[-1].split('2milmoses_')[1].split('.json')[0].split('_')[0]
        labels.append(fn)

    data = {}
    for fn, label in zip(fns, labels):
        with open(fn, 'r') as f:
            dump = json.load(f)
        data[label] = dump
    return data, labels


def make_model_name(_prop, model_number, semi_sup_percent):
    
    if   _prop == "bch":
        prop="bch"
    elif _prop == "bc":
        prop="boman-chargepH7p2"
    elif _prop == "b":
        prop="boman"
    elif _prop == "c":
        prop="chargepH7p2"
    elif _prop == "h":
        prop="hydrophobicity"
    elif _prop in ["predicted-log10mic", "oracle"]:
        prop = "predicted-log10mic"
    
    if (semi_sup_percent==100 or semi_sup_percent=="100"):
        percent = ""
        suffix="dPP64-ZScore"
    elif (semi_sup_percent==0 or semi_sup_percent=="0"):
        if prop=="predicted-log10mic":
            percent = "0-"
        else:
            percent = ""
        suffix="cdhit90-zScoreNormalized"
    else:
        percent = str(semi_sup_percent)+"-"
        suffix="cdhit90-zScoreNormalized"

    model_name=f"transvae-64-peptides-{prop}-zScoreNormalized-{percent}organized-{suffix}"

    return model_name

def get_latent_spaces():
    mu_embedding_files = []
    data_dir = "analysis/"
    file_to_grab_prefix = "mu_transvae-64"
    file_to_grab_suffix = "train.npy"
    
    for _prop in ['boman', 'hydrophobicity', 'chargepH7p2', 'boman-chargepH7p2', 'bch', "predicted-log10mic"]:
        percentages = [0,25,50,75, 98]
        if _prop=="predicted-log10mic":
            percentages = [0, 98]
        for _perc in percentages:
            
            for _d in os.listdir(data_dir):
                
                
                condn_dir = (os.path.isdir(data_dir+_d))
                if _perc==0:
                    if _prop=="predicted-log10mic":
                        condn_prefix = (_d.startswith(f"transvae-64-peptides-{_prop}-zScoreNormalized-0-org"))
                    else:
                        condn_prefix = (_d.startswith(f"transvae-64-peptides-{_prop}-zScoreNormalized-org"))
                else:
                    condn_prefix = (_d.startswith(f"transvae-64-peptides-{_prop}-zScoreNormalized-{_perc}"))
                condn_suffix = (_d.endswith("organized-cdhit90-zScoreNormalized") )
                condn_not = ("--" not in _d)
                
                if condn_dir and condn_prefix and condn_suffix and condn_not:
                    print(f"{_d=}")
                    for _f in os.listdir(data_dir+_d):
                        if _f.startswith(file_to_grab_prefix) and _f.endswith(file_to_grab_suffix):
                            print(f"{_f=}")
                            mu_embedding_files.append(os.path.join(_d, _f) )
                    print()
    
    latent_spaces = {}
    file_to_grab_prefix = "mu_transvae-64"
    file_to_grab_suffix = "train.npy"
    
    for _prop in ['boman', 'hydrophobicity', 'chargepH7p2', 'boman-chargepH7p2', 'bch', "predicted-log10mic"]:
        percentages = [0,25,50,75, 98]
        if _prop=="predicted-log10mic":
            percentages = [0, 98]
            
        for _perc in percentages:
            for _dir_file in mu_embedding_files:
                _d, _f = _dir_file.split("/")
                if _perc==0:
                    if _prop=="predicted-log10mic":
                        if _d.endswith("zScoreNormalized-0-organized-cdhit90-zScoreNormalized") and (f"peptides-{_prop}-z" in _d ):
                                print(_d)
                                latent_spaces[_prop+"-"+str(_perc)] = np.load(os.path.join(data_dir,_dir_file) )
                    else:
                        if _d.endswith("zScoreNormalized-organized-cdhit90-zScoreNormalized") and (f"peptides-{_prop}-z" in _d ):
                            print(_d)
                            latent_spaces[_prop+"-"+str(_perc)] = np.load(os.path.join(data_dir,_dir_file))
                else:
                    if (_prop+"-zS" in _d) and (f"-{_perc}-org" in _d):
                        
                        latent_spaces[_prop+"-"+str(_perc)] = np.load(os.path.join(data_dir,_dir_file))
    
    return latent_spaces


def get_boloop_runs(model_name, perc, dim_reduction_method, box_bounds, n_pca_dims=5):

    if box_bounds==10:
        box_bound_info = f"_neg{box_bounds}to{box_bounds}_"
    else:
        box_bound_info = ""

    if n_pca_dims==5:
        pca_dim_info = ""
    else:
        pca_dim_info = f"_PCAdims{n_pca_dims}_"
    
    if (perc==0 or perc=="0") and ("predicted-log10mic" not in model_name):
        _fname = f"boloop_results_{dim_reduction_method}{box_bound_info}{pca_dim_info}{model_name}_log10_mic.pkl"
        print("in first if")
        print(f"{_fname=}")
        
        with open(f"analysis/{model_name}/boloop_results/{_fname}", 'rb') as f:
            data=pkl.load(f)
        
        runs = [data[f'run_{j}'] for j in range(5)]
    else:
        # check if boloop_results file exists first
        _fname = f"boloop_results_{dim_reduction_method}{box_bound_info}{pca_dim_info}{model_name}_log10_mic.pkl"
        print(f"{_fname=}")
        if os.path.exists(f"analysis/{model_name}/boloop_results/{_fname}"):
            with open(f"analysis/{model_name}/boloop_results/{_fname}", 'rb') as f:
                data=pkl.load(f)
            
            runs = [data[f'run_{j}'] for j in range(5)]        
        else:
            print("boloop results file doesn't exist?")
            runs = []
            for i in range(5):
                if "predicted-log10mic" in model_name:                    
                    _fname = f"optimization_results_{model_name}_run{i}.pkl"
                else:
                    _fname = f"optimization_results_{dim_reduction_method}{box_bound_info}{model_name}_run{i}.pkl"
                print(f"{_fname=}")
                _fpath = f"analysis/{model_name}/boloop_results/{_fname}"
                if os.path.exists(_fpath):
                    with open(_fpath, "rb") as f:
                        _run=pkl.load(f)
                elif dim_reduction_method=="PCA":
                    _fname = f"optimization_results_{model_name}_run{i}.pkl"
                    print(f"trying filename: {_fname}")
                    _fpath = f"analysis/{model_name}/boloop_results/{_fname}"
                    with open(_fpath, "rb") as f:
                        _run=pkl.load(f)
                        
                runs.append( _run )
    
    return runs

#
def get_top_2_most_correlated_PCs(df, property_col, return_pcs=True):
    """
    Computes the pearson correlation coefficient (PCC) between principal components 
    and the desired property (property_col). Returns the top two, or returns all
    with their PCCs

    Parameters:
    ----------
    df: pd.DataFrame
        the data. Columns like: [pc1, pc2, pc3, ..., pcn, property_col]

    property_col: str
        The column name in the df for the property we compute correlation with.

    Returns:
    --------
        Top two principal components,
        OR
        A sorted list of tuples (pci, pcci), the principal component and its 
        corresponding correlation value. 
    """
    
    _prop_values = df[property_col].values

    pcs_and_corrs = []
    for _col in df.columns:
        if _col.startswith("pc"):
            correl = pearsonr(df[_col].values, _prop_values)
            pcs_and_corrs.append( (_col, abs(correl.statistic)) )

    pcs_and_corrs = sorted(pcs_and_corrs, key= lambda x: x[1], reverse=True)

    if return_pcs:
        return pcs_and_corrs[0][0], pcs_and_corrs[1][0] 
    else:
        return pcs_and_corrs