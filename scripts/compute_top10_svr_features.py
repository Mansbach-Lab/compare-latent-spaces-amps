

import numpy as np
import pandas as pd
import os
import pickle as pkl

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from propy import PyPro


feature_keys = [
    'tausw2', 'PolarizabilityC3', 'ChargeT12', 'NormalizedVDWVC2',
    'PAAC20', 'QSOSW39', 'PAAC21', 'MoreauBrotoAuto_Steric2',
    'ChargeD2100', 'MoranAuto_Hydrophobicity5'
]

def compute_selected_features(sequence: str) -> dict:
    DesObject = PyPro.GetProDes(sequence)
    features = {}

    features["tausw2"] = DesObject.GetSOCN()['tausw2']
    features["PolarizabilityC3"] = DesObject.GetCTD()['_PolarizabilityC3']
    features["ChargeT12"] = DesObject.GetCTD()['_ChargeT12']
    features["NormalizedVDWVC2"] = DesObject.GetCTD()['_NormalizedVDWVC2']
    features["PAAC20"] = DesObject.GetPAAC(lamda=4)['PAAC20']
    features["QSOSW39"] = DesObject.GetQSO()['QSOSW39']
    features["PAAC21"] = DesObject.GetPAAC(lamda=4)['PAAC21']
    features["MoreauBrotoAuto_Steric2"] = DesObject.GetMoreauBrotoAuto()['MoreauBrotoAuto_Steric2']
    features["ChargeD2100"] = DesObject.GetCTD()['_ChargeD2100']
    features["MoranAuto_Hydrophobicity5"] = DesObject.GetMoranAuto()['MoranAuto_Hydrophobicity5']

    return features

def process_sequence(i, sequence):
    result = {'sequence': sequence}
    try:
        features = compute_selected_features(sequence)
        for key in feature_keys:
            result[key] = features.get(key, np.nan)
    except Exception as e:
        # print(f"failed on sequence = {i} {sequence}\nerror: {e}")
        for key in feature_keys:
            result[key] = np.nan
    return result


if __name__=="__main__":
    print("starting to compute top10 SVR features of every peptide in training set")
    predicted_log10mic_train = pd.read_csv("data/peptides_predicted-log10mic_zScoreNormalized_train.txt")
    predicted_log10mic_test  = pd.read_csv("data/peptides_predicted-log10mic_zScoreNormalized_test.txt")
    
    properties_train = pd.read_csv("data/properties/peptides_2024_cdhit90_unbalanced_train_properties_zScoreNormalized.txt")

    nan_mask = (~predicted_log10mic_train.predicted_mic.isna())

    sequences = pd.read_csv("data/peptides_2024_cdhit90_unbalanced_train.txt")
    
    # Parallel execution
    results_list = Parallel(n_jobs=10)(
        delayed(process_sequence)(i, seq) for i, seq in enumerate(sequences.peptides)
    )
    
    # Combine results
    results = {key: [r[key] for r in results_list] for key in ['sequence'] + feature_keys}


    with open("data/top10_predict_mic_features.pkl",'wb') as f:
        pkl.dump(results, f)