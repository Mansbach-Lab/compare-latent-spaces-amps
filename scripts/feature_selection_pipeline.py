import pandas as pd
import numpy as np
import scipy.optimize as scop

import peptides
import propy

import matplotlib.pyplot as plt

import argparse
import pickle as pkl
import os
import time

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit

from sklearn.feature_selection import mutual_info_regression

from transvae.mic_svr import compute_propy_properties, VS_SSVR, perform_mRMR, perform_lasso

def load_data(data_dir, data_type="mic"):

    # load data as used by Witten & Witten (2019)
    if data_type=="mic":
        propy_des = pd.read_csv(data_dir+"train_propy_des.csv")

        ecoli_train_with_c = pd.read_pickle(f'{data_dir}ecoli_train_with_c_df.pkl')
        train_Y = ecoli_train_with_c.value
    elif data_type=="hemolytik":
        propy_des = pd.read_csv(data_dir+"train_propy_des_hemolytik.csv")

        hemolytik_train = pd.read_csv(data_dir+"train_log10_HC50.csv")
        train_Y = hemolytik_train.log10_HC50
    else:
        raise ValueError("data_type must be either 'mic' or 'hemolytik'")
    

    input_data = propy_des.dropna()
    train_Y = train_Y[(propy_des.isna().sum(axis=1)==0).values]

    train_Y = train_Y.loc[
        ((input_data.sequence.str.len()>4) &
        (input_data.sequence.str.len()<101)).values
    ]

    input_data = input_data.loc[
        (input_data.sequence.str.len()>4) &
        (input_data.sequence.str.len()<101)
    ]
    input_data = input_data.loc[:,input_data.columns[1:]]

    return input_data, train_Y

def main(data_dir, N_CPUS=1, data_type="mic", do_mrmr=True, do_model_feature_selection=False):

    # gets training and testing data, as used by Witten & Witten (2019). 

    if os.getcwd().split("/")[-1] == "scripts":
        data_dir = "../oracles/"
    else:
        data_dir = "oracles/"

    # load data
    input_data, train_Y = load_data(data_dir, data_type=data_type)

    ########################################
    # perform minimum redundancy maximum relevance feature selection/filtering
    ########################################
    t0 = time.time()
    results, relevancies, pairwise_redundancies = perform_mRMR(
        input_data, train_Y, N_CPUS=N_CPUS, max_features=input_data.shape[1], verbose=True
    )
    
    print(f"{time.time()-t0}s")

    print("pickling results")
    mrmr_results = {
        "results": results,
        "relevancies": relevancies,
        "pairwise_redundancies": pairwise_redundancies
    }
    with open(data_dir+f"mrmr_results_{data_type}.pkl", "wb") as f:
        pkl.dump(mrmr_results, f)
    print("done mRMR")

    # get the best subset of features
    best_subset = None
    for _feature_subset in list(mrmr_results['results'].keys()):
        _inc_mrmr_score = mrmr_results['results'][ _feature_subset ]["incremental_mRMR_score"]
        if _inc_mrmr_score>0:
            best_subset = _feature_subset

    propy_features_post_mrmr = []
    for i in best_subset:
        propy_features_post_mrmr.append(list(input_data.columns)[i])

    ########################################
    # perform model-based feature selection using either LASSO or sparse SVC
    ########################################
    if do_model_feature_selection:
        input_data_post_mrmr = input_data[propy_features_post_mrmr]
        model_feature_selection_results, sorted_features_df = perform_lasso(
            input_data_post_mrmr, train_Y, N_CPUS=None, method="cv", verbose=False
        )


if __name__ == "__main__":
    # take data directory and number of cpus as command-line arguments
    parser = argparse.ArgumentParser(description="Run feature selection pipeline")
    parser.add_argument("--data_dir", type=str, help="path to the data directory")
    parser.add_argument("--n_cpus", type=int, help="number of cpus to use (only applicable to mRMR atm)")
    parser.add_argument("--data_type", type=str, help="data to use (mic or hemolytik)")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    N_CPUS = args.n_cpus
    data_type = args.data_type

    # check that data_type is in ["mic", "hemolytik"]
    if data_type not in ["mic", "hemolytik"]:
        raise ValueError("data_type must be either 'mic' or 'hemolytik'")
    
    print(f"{data_dir=}, {N_CPUS=}, {data_type=}")

    main(data_dir, N_CPUS=N_CPUS, data_type=data_type)