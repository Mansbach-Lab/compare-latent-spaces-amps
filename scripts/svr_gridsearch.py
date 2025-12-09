import numpy as np
import pandas as pd
import time
import argparse
import pickle as pkl

from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

def evaluate_params(c_, epsilon_, X, y, important_feature_cols, shuffler, scaler, n_splits, kernel='rbf'):
    svr_ = SVR(kernel=kernel, C=c_, epsilon=epsilon_)
    avg_mse_validation_ = 0
    results = {
        'C_values':[],
        'epsilon_values':[],
        'mse_training':[],
        'sse_validation':[],
        "r2_training":[],
        "r2_validation":[]
    }
    for train_index, test_index in shuffler.split(X):
        N=len(train_index)
        
        train_X_ = X.iloc[train_index]
        train_Y_ = y.iloc[train_index]

        test_X_ = X.iloc[test_index]
        test_Y_ = y.iloc[test_index]

        # center features
        scaler.fit(train_X_)
        train_X_ = scaler.transform(train_X_)
        test_X_  = scaler.transform(test_X_)

        # fit
        svr_.fit(train_X_, train_Y_)

        # compute MSE
        mse_training_ = mean_squared_error(train_Y_, svr_.predict(train_X_))
        r2_training_ = svr_.score(train_X_, train_Y_)

        sse_validation_ = N*mean_squared_error(test_Y_, svr_.predict(test_X_))
        avg_mse_validation_ += sse_validation_
        r2_validation_ = svr_.score(test_X_, test_Y_)

        results['C_values'].append(c_)
        results['epsilon_values'].append(epsilon_)
        results['mse_training'].append(mse_training_)
        results['sse_validation'].append(sse_validation_)
        results['r2_training'].append(r2_training_)
        results['r2_validation'].append(r2_validation_)

    avg_mse_validation_ /= (n_splits*N)
    return results, avg_mse_validation_

def coarse_hyperparameter_sweep_par(X, y, important_feature_cols, N_CPUS=1, n_splits=10):
    shuffler = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    # 45x45 grid search with 20 cpus will take about 1.5hrs
    logC_values = np.linspace(-2, 2, 45)
    logeps_values = np.linspace(-2, 2, 45)
    C_values = np.power(10, logC_values)
    epsilon_values = np.power(10, logeps_values)
    
    results = {
        'C_values':[],
        'epsilon_values':[],
        'mse_training':[],
        'sse_validation':[],
        "r2_training":[],
        "r2_validation":[],
        'best_params':{'C':None, 'epsilon':None},
        'best_mse':np.inf
    }

    param_combinations = [(c_, epsilon_) for c_ in C_values for epsilon_ in epsilon_values]
    
    parallel_results = Parallel(n_jobs=N_CPUS)(
        delayed(evaluate_params)(c_, epsilon_, X, y, important_feature_cols, shuffler, scaler, n_splits)
        for c_, epsilon_ in param_combinations
    )

    for res, avg_mse_validation_ in parallel_results:
        for key in res:
            results[key].extend(res[key])
        if avg_mse_validation_ < results['best_mse']:
            results['best_mse'] = avg_mse_validation_
            results['best_params']['C'] = res['C_values'][0]
            results['best_params']['epsilon'] = res['epsilon_values'][0]

    return results


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='SVR hyperparameter search')

    parser.add_argument('--out_dir', type=str, default='data/', help='output directory')
    parser.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument("--train_X", type=str, help="path to the input data")
    parser.add_argument("--train_Y", type=str, help="path to the training labels/target values")
    parser.add_argument("--important_features", type=str, help="path to the list of important features")
    parser.add_argument("--data_type", type=str, choices=['mic','hemolytik'], help="data to use (mic or hemolytik)")
    
    args = parser.parse_args()
    
    out_dir = args.out_dir
    N_CPUS   = args.n_cpus
    input_data = pd.read_csv(args.train_X)
    train_Y    = pd.read_csv(args.train_Y)
    feature_weight_nonzero_df = pd.read_csv(args.important_features, keep_default_na=False) # "notsorted_features_nonzero_log10_mic.csv"

    input_data_important_features = input_data[feature_weight_nonzero_df.feature_name]

    important_feature_cols = list(feature_weight_nonzero_df.feature_name)
    results = coarse_hyperparameter_sweep_par(input_data_important_features, train_Y, important_feature_cols, N_CPUS=N_CPUS, n_splits=10)    
    
    with open(f'{out_dir}svr_gridsearch_results_{args.data_type}.pkl', 'wb') as f:
        pkl.dump(results, f)
    pass