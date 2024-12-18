import pandas as pd
import numpy as np
import torch
import pickle as pkl
import logging
import time
import argparse
import sys

from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from transvae.transformer_models import TransVAE
from transvae.tvae_util import *

from transvae.optimization import OptimizeInReducedLatentSpace
from transvae.mic_svr import NonlinearSVRonPhysicoChemicalProps

def encode_seqs(df, stoi):
    encoded_seqs = []
    max_len = 101
    for seq in df['sequence']:
        temp_ = [stoi[aa] for aa in seq]
        seq_len = len(seq)
        temp_ += [stoi["<end>"]]
        temp_ += [stoi["_"]] * (max_len - seq_len)
        temp_ = [0] + temp_
        encoded_seqs.append(temp_)
    df = pd.DataFrame({"encoded_sequence":encoded_seqs})
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

def make_nonlinear_svr(prediction_target:str):
    logging.info(f"Building SVR for {prediction_target}")
    if prediction_target not in ['log10_mic', 'log10_HC50']:
        raise ValueError(f"Invalid prediction_target: {prediction_target}. Must be one of ['log10_mic', 'log10_HC50']")
    
    input_data  = pd.read_csv(f"oracles/train_X_input_data_{prediction_target}.csv")
    train_Y_mic = pd.read_csv(f"oracles/train_Y_{prediction_target}.csv")
    feature_weight_nonzero_df = pd.read_csv(f"oracles/notsorted_features_nonzero_{prediction_target}.csv", keep_default_na=False)
    
    if prediction_target == 'log10_mic':
        svr_gridsearch_results_fpath = "oracles/svr_gridsearch_results_mic.pkl"
    elif prediction_target == 'log10_HC50':
        svr_gridsearch_results_fpath = "oracles/svr_gridsearch_results_hemolytik.pkl"
    with open(svr_gridsearch_results_fpath, 'rb') as f:
        svr_gridsearch_results = pkl.load(f)
    
    important_feature_cols = list(feature_weight_nonzero_df.feature_name)
    
    input_data_important_features = input_data[feature_weight_nonzero_df.feature_name]
    
    nonlinear_svr = NonlinearSVRonPhysicoChemicalProps(prediction_target,
                                       important_feature_cols,
                                             C=svr_gridsearch_results['best_params']['C'],
                                       epsilon=svr_gridsearch_results['best_params']['epsilon']
                  )
    nonlinear_svr.fit(input_data_important_features, train_Y_mic, input_type='physicochemical_properties')

    return nonlinear_svr

def run_optimization(i, params, model, nonlinear_svr, pca, char_dict, minimize_or_maximize, train_X, train_Y):
    logging.info(f"Run {i+1}/{params['n_bo_runs']}")
    dupl_params = params.copy()
    dupl_params["run"] = i
    # Initialize optimizer
    _optimizer = OptimizeInReducedLatentSpace(
        model, nonlinear_svr, pca, char_dict, minimize_or_maximize_score=minimize_or_maximize, params=dupl_params
    )

    # Run optimization loop
    _optimizer.optimize(train_X, train_Y, n_iters=params['n_bo_iters'], n_restarts=1,verbose=False)

    # Get results
    return f"run_{i}", _optimizer.optimization_results

def parallel_optimization(params, model, nonlinear_svr, dimensionality_reducers, char_dict, minimize_or_maximize, train_Xs, train_Ys, N_CPUS):
    
    if params['n_different_initializations'] == 1:
        results = dict(Parallel(n_jobs=N_CPUS)(
            delayed(run_optimization)(i, params, model, nonlinear_svr, dimensionality_reducers[0], char_dict, minimize_or_maximize, train_Xs[0], train_Ys[0]) for i in range(params['n_bo_runs'])
            )
        )
        return results
    else:    
        results = dict(Parallel(n_jobs=N_CPUS)(
            delayed(run_optimization)(i, params, model, nonlinear_svr, dimensionality_reducers[i], char_dict, minimize_or_maximize, train_Xs[i], train_Ys[i]) for i in range(params['n_bo_runs'])
            )
        )
        return results

def main(data_X, data_Y, params):


    #########################################
    # load a trained generative model
    logging.info("loading generative model...")
    if not params['use_esm']:
        model_src = params['chkpt_fpath']
        model_obj=torch.load(model_src, map_location=torch.device("cpu"))
        model = TransVAE(load_fn=model_src, workaround="cpu")
        
        model.params['HARDWARE']= 'cpu'
        model.params["BATCH_SIZE"] = N_INITALIZATION_POINTS
        print(data_X[:5])
        with torch.no_grad():
            z, mu, logvar = model.calc_mems(data_X.to_numpy(), log=False,save=False)
    else:
        print("Using ESM model")
        print(f"{data_X.shape[0]} sequences, {data_X[:5]}")
        print(list(data_X.to_numpy().flatten()))
        logging.info("loading ESM model...")
        model = EsmWrapper(params)
        with torch.no_grad():
            mu = model.encode(list(data_X.to_numpy().flatten()[:2]))
        print(f"mu shape: {mu.shape}")
        sys.exit()

    #########################################
    # build a dimensionality reduction method
    logging.info("building PCA w/ {N_PCA_COMPONENTS} components...")
    dimensionality_reducers = []
    train_Xs = []
    train_Ys = []
    for i in range(params['n_different_initializations']):
        _pca = PCA(n_components=N_PCA_COMPONENTS)
            
        _pca.fit(mu[i*params['n_initialization_points']:(i+1)*params['n_initialization_points']])
        _pca_mu = _pca.transform(
            mu[i*params['n_initialization_points']:(i+1)*params['n_initialization_points']]
        )
        dimensionality_reducers.append(_pca)

        # 
        if isinstance(_pca_mu, np.ndarray):
            train_X = torch.from_numpy(_pca_mu)
        elif isinstance(_pca_mu, pd.DataFrame):
            train_X = torch.from_numpy(_pca_mu.values)
        elif isinstance(_pca_mu, pd.Series):
            train_X = torch.from_numpy(_pca_mu.values)
        
        train_Xs.append(train_X)

        train_Y = data_Y[i*params['n_initialization_points']:(i+1)*params['n_initialization_points']]
        if isinstance(train_Y, np.ndarray):
            train_Y = torch.from_numpy(train_Y)
        elif isinstance(train_Y, pd.DataFrame):
            train_Y = torch.from_numpy(train_Y.values)
        elif isinstance(train_Y, pd.Series):
            train_Y = torch.from_numpy(train_Y.values)

        if train_Y.dim() == 1:
            train_Y.unsqueeze_(1)
        train_Ys.append(train_Y)

    #########################################
    # build a nonlinear SVR ORACLE
    logging.info("building oracle...")
    if params['use_mic_svr']:
        nonlinear_svr = make_nonlinear_svr( params['prediction_target'] )
    else:
        raise NotImplementedError("Only mic svr is supported at the moment")
    
    #########################################
    # perform optimization
    logging.info(f"Running BO for {params['n_bo_runs']} runs, each with {params['n_bo_iters']} oracle calls/BO loop iterations...")

    minimize_or_maximize = "minimize" if params["prediction_target"]=="log10_mic" else "maximize"
    results = parallel_optimization(params, model, nonlinear_svr, dimensionality_reducers, char_dict, minimize_or_maximize, train_Xs, train_Ys, params['N_CPUS'])

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',        type=str, required=True)
    parser.add_argument('--seq2prop_fpath',  type=str, required=True)
    parser.add_argument('--char_dict_fpath', type=str, required=True)
    parser.add_argument('--chkpt_fpath',     type=str, required=True)
    parser.add_argument('--output_dir',      type=str, required=False, default='./')
    
    parser.add_argument("--use_esm", type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--use_mic_svr', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--n_bo_runs',               type=int, default=  5)
    parser.add_argument('--n_bo_iters',              type=int, default=100)
    parser.add_argument('--n_initialization_points', type=int, default=100)
    parser.add_argument('--n_different_initializations', type=int, default=1)
    parser.add_argument('--n_pca_components',        type=int, default=  5)
    parser.add_argument('--prediction_target', type=str, default='log10_mic', choices=['log10_mic', 'log10_HC50'])
    parser.add_argument('--n_cpus', type=int, default=1)

    args = parser.parse_args()

    USE_MIC_SVR = args.use_mic_svr
    if USE_MIC_SVR == 'yes':
        USE_MIC_SVR = True
    else:
        USE_MIC_SVR = False
    USE_Amplify = not USE_MIC_SVR

    USE_ESM = args.use_esm
    if USE_ESM == 'yes':
        USE_ESM = True

        # import relevant module
        from transvae.helpers_esm import EsmWrapper

        # Define model paths
        model_path = "esm2/esm_files/"
        model_name_or_path     = model_path
        tokenizer_name_or_path = model_path

        # logging.info("Loading model...")
        # model = EsmForProteinFolding.from_pretrained(
        #                     model_path,
        #                     local_files_only=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #                     model_path,
        #                     local_files_only=True
        # )

    else:
        USE_ESM = False

    N_BO_RUNS              = args.n_bo_runs
    N_BO_ITERS             = args.n_bo_iters
    N_INITALIZATION_POINTS = args.n_initialization_points
    N_DIFFERENT_INITIALIZATIONS = args.n_different_initializations
    N_PCA_COMPONENTS       = args.n_pca_components
    N_CPUS                 = args.n_cpus

    data_dir   = args.data_dir
    output_dir = args.output_dir
    seq2prop_fpath  = args.seq2prop_fpath
    char_dict_fpath = args.char_dict_fpath
    chkpt_fpath = args.chkpt_fpath

    with open(char_dict_fpath, 'rb') as f:
        char_dict = pkl.load(f)

    #########################################
    # set up logging config
    logfilename = f'{data_dir}bayesian_optimization_loop.log'
    if USE_ESM:
        logfilename = f'{data_dir}bayesian_optimization_loop_esm.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfilename)

    #########################################
    # load data
    train_seqs_to_prop = pd.read_csv(seq2prop_fpath)
    
    sampled_seqs_to_prop = train_seqs_to_prop.sample(N_INITALIZATION_POINTS*N_DIFFERENT_INITIALIZATIONS, 
                                                     ignore_index=True, 
                                                     random_state=42
    )
    
    sampled_seqs  = sampled_seqs_to_prop[['sequence']]
    sampled_props = sampled_seqs_to_prop[args.prediction_target] # LEAST GENERAL PART OF THE CODE
    sampled_seqs_enc = encode_seqs(sampled_seqs, char_dict)

    params = {"use_mic_svr":USE_MIC_SVR,}
    params['n_bo_iters'] = N_BO_ITERS
    params['n_bo_runs'] = N_BO_RUNS
    params['n_initialization_points'] = N_INITALIZATION_POINTS
    params['n_different_initializations'] = N_DIFFERENT_INITIALIZATIONS
    params['n_pca_components'] = N_PCA_COMPONENTS
    params['prediction_target'] = args.prediction_target
    
    params["chkpt_fpath"] = chkpt_fpath
    params['char_dict'] = char_dict
    params['output_dir'] = output_dir
    
    params["use_esm"] = USE_ESM
    if USE_ESM:
        params["esm_path"] = model_path
    
    params['N_CPUS'] = N_CPUS

    #########################################
    # run main
    #########################################
    t0 = time.time()
    results = main(sampled_seqs, sampled_props, params)
    print(f"Elapsed time: {time.time()-t0:.4f} seconds. For {N_BO_RUNS} runs, each with {N_BO_ITERS} oracle calls/BO loop iterations.")

    #########################################
    # save results
    output_fpath = f"{output_dir}bayesian_optimization_results_{chkpt_fpath.split('/')[1]}_{params['prediction_target']}.pkl"
    if USE_ESM:
        output_fpath = output_fpath.replace(".pkl", "_esm.pkl")
    results['params'] = params
    with open(output_fpath, 'wb') as f:
        pkl.dump(results, f)