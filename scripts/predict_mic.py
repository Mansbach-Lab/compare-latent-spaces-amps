import pandas as pd
import numpy as np
import torch
import pickle as pkl

import time
import argparse
import sys
import os

import seaborn as sns

from joblib import Parallel, delayed

from sklearn.decomposition import PCA

from transvae.mic_svr import NonlinearSVRonPhysicoChemicalProps

def make_nonlinear_svr(prediction_target:str):
    print("making nonlinear svr for ",prediction_target)
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

def predict_mic_fn(x, nonlinear_svr):
    try:
        return nonlinear_svr.predict(x)[0]
    except TypeError:
        return np.nan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SVR on physicochemical properties')
    parser.add_argument('--prediction_target', type=str, help='Target to predict', required=False, default='log10_mic')
    args = parser.parse_args()
    
    _do_train_set = False
    _do_test_set  = True

    nonlinear_svr = make_nonlinear_svr(args.prediction_target)
    
    if _do_train_set:
        df_train = pd.read_csv("data/peptides_2024_cdhit90_unbalanced_train.txt")
        print(df_train.head() )

        print("Now predict MICs for train set:")
        predicted_mics = []
        countNaNs=0
        t0=time.time()
    
        N_CPUs = 8
        chunk_size = len(df_train) // N_CPUs
        chunks = [df_train.iloc[i : i + chunk_size, 0] for i in range(0, len(df_train), chunk_size)]
    
        if len(chunks)!=N_CPUs:
            raise ValueError(f"{len(chunks)=} != {N_CPUs=}")
    
        print("predicting training set..")

        predicted_mics = Parallel(n_jobs=N_CPUs)(
            delayed(lambda chunk: [predict_mic_fn(x, nonlinear_svr) for x in chunk])(chunk) for chunk in chunks
        )
    
        print("predicting training set done. now saving..")

        # now re-organize them back together
        reorganized_predicted = []
        for _chunk in predicted_mics:
            reorganized_predicted += _chunk

        df_train['predicted_mic'] = reorganized_predicted
        df_train.to_csv(f"data/peptides_predicted_{args.prediction_target}_train.txt", index=False)

    if _do_test_set:
        print("Now predict MICs for test set:")
        df_test = pd.read_csv("data/peptides_2024_cdhit90_unbalanced_test.txt")
    
        N_CPUs = 6
        chunk_size = len(df_test) // N_CPUs
        chunks = [df_test.iloc[i : i + chunk_size, 0] for i in range(0, len(df_test), chunk_size)]
    
        predicted_mics = Parallel(n_jobs=N_CPUs)(
            delayed(lambda chunk: [predict_mic_fn(x, nonlinear_svr) for x in chunk])(chunk) for chunk in chunks
        )
    
        print("finished predicting test set. now saving...")

        reorganized_predicted = []
        for _chunk in predicted_mics:
            reorganized_predicted += _chunk

        df_test['predicted_mic'] = reorganized_predicted
        df_test.to_csv(f"data/peptides_predicted_{args.prediction_target}_test.txt", index=False)
