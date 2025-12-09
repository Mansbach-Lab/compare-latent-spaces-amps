import pandas as pd
import numpy as np
import pickle as pkl
import scipy
import peptides
import propy
import os
import time 
import logging

from propy import PyPro
from propy.PyPro import GetProDes
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from joblib import Parallel, delayed

def make_svr(final_mic_svr_path, feature_selection_results_path,use_cysteinic_data=False, return_train_data=False):
    """
    Fits and returns a Support Vector Regression model
    on the physicochemical properties of peptide sequences.
    Data is taken from Witten & Witten (2019).
    """
    raise DeprecationWarning("This function is deprecated. Use the NonlinearSVRonPhysicoChemicalProps class instead.")
    # gets training and testing data, as used by Witten & Witten (2019). 
    cwd = os.getcwd()


    if return_train_data:
        return svr, train_X, train_Y
    else:
        return svr

def compute_propy_properties(sequences:list[str]):
    """
    Computes the physicochemical properties of a peptide sequence.
    Arguments:
    ----------
    sequences: list[str], protein sequences

    Returns:
    --------
    pd.DataFrame, dataframe containing the physicochemical properties of the sequences
    """
    if not isinstance(sequences,list):
        raise ValueError("input data type must be a list of strings")
    if not all([isinstance(seq_,str) for seq_ in sequences]):
        raise ValueError("all elements in the list must be strings")
    
    results = {"sequence":sequences}
    min_len = min([len(seq) for seq in sequences])
    max_len = max([len(seq) for seq in sequences])
    lambda_max = max(4, min_len-1)
    for i, seq in enumerate(sequences):
        
        succeeded = False
        # print(seq, len(seq))
        Des = GetProDes(seq)
        if len(seq)<11:
            paac_lbda_  = len(seq)-1
            apaac_lbda_ = len(seq)-1
        else:
            paac_lbda_ = 10
            apaac_lbda_ = 10
        
        if len(seq)<5:
            succeeded = False
        else:
            try:
                alldes_ = Des.GetALL(
                    paac_lamda =lambda_max,
                    apaac_lamda=lambda_max,
                )
                succeeded = True
            except:
                print(f"failed with seq={seq}, {len(seq)=}")
                succeeded = False

        if succeeded:
            if i==0:
                for key_ in alldes_:
                    results[key_] = [alldes_[key_]]
            else:
                for key_ in alldes_:
                    results[key_].append(alldes_[key_])
        else:
            # if propy was not able to compute properties, fill with np.nan
            for key_ in results:
                if key_=="sequence":
                    continue
                else:
                    results[key_].append(np.nan)
        
    return pd.DataFrame(results)

def calculate_mutual_info(i, data):
    mutual_info_ = mutual_info_regression(data, data[:, i], n_neighbors=3, random_state=42)
    return mutual_info_

def perform_mRMR(data:pd.DataFrame, targets:pd.DataFrame, N_CPUS=1, method="mutual_info", max_features = 4, verbose=False):
    """
    Performs minimum Redundancy Maximum Relevance feature selection on the data.
    Note: 
        - redundancy computation with mutual_info takes a long time. 
            check computation time of a couple features with all, before running all against all.
    Arguments:
    ----------
    data: pd.DataFrame, input data
    targets: pd.DataFrame, target values
    N_CPUS: int, number of cpus available for parallel computation
    method: str, choices=["mutual_info","f_classif","f_regression", "correlation"]
            method for computing the relevance and redundancy of features
    
    Returns:
    --------
    outputs: dict, dictionary containing the following keys:
        tuple(S) : dict, dictionary containing the following
            "relevance"  : float, relevance  of the feature set
            "redundancy" : float, redundancy of the feature set
            "mRMR_score" : float, mRMR score of the feature set
        (S is a list of feature indices) 
    """
    # max_features = min(4, data.shape[1])
    ############################
    # compute the relevance of each feature
    if method=="mutual_info":
        t0 = time.time()
        relevances = mutual_info_regression(data, 
                                            targets, 
                                            n_neighbors=3,
                                            random_state=42, 
        )
        if verbose:
            print(f"Computing relevancy for {max_features} features took {round(time.time()-t0,4)} seconds")
    elif method=="f_classif":
        relevances = f_classif(data, targets)
    elif method=="f_regression":
        relevances = f_regression(data, targets)
    elif method=="correlation":
        relevances = data.corrwith(targets)
    else:
        raise ValueError("method must be one of ['mutual_info','f_classif','f_regression','correlation']")
    
    ############################
    # compute the redundancy of each feature to each other feature
    pairwise_redundancies = np.zeros((data.shape[1],data.shape[1]))
    # for i in range(data.shape[1]):
    t0 = time.time()
    if N_CPUS>1:
        results = Parallel(n_jobs=N_CPUS)(delayed(calculate_mutual_info)(i, data.values) for i in range(max_features))
        for i, mutual_info_ in enumerate(results):
            pairwise_redundancies[i] = mutual_info_
    else:
        for i in range(max_features):
            mutual_info_ = mutual_info_regression(data, 
                                                data.iloc[:,i], 
                                                n_neighbors=3, 
                                                random_state=42)
            
            pairwise_redundancies[i] = mutual_info_
    if verbose:
        print(f"Computing pairwise redundancies for {max_features} features took {round(time.time()-t0,4)} seconds")
    ############################
    # compute mRMR scores

    # find first feature that maximizes the relevance score    
    S = []
    max_j = np.argmax(relevances)
    S.append(max_j)
    outputs = {}
    outputs[tuple(S)] = {"incremental_relevance":relevances[max_j], 
                         "redundancy":0.0,
                         "incremental_mRMR_score":relevances[max_j]}

    # perform incremental search
    t0 = time.time()
    for i in range(max_features):
        # find the feature that maximizes the mRMR score
        max_score = -np.inf
        max_j = None
        # for j in range(data.shape[1]):
        for j in range(max_features):
            if j in S:
                continue
            relevance_  = relevances[j]
            redundancy_ = np.mean(pairwise_redundancies[j][S])
            score_ = relevance_ - redundancy_
            if score_ > max_score:
                max_score = score_
                max_rel = relevance_
                min_red = redundancy_
                max_j = j
        S.append(max_j)
        outputs[tuple(S)] = {"incremental_relevance":max_rel, 
                             "redundancy":min_red,
                             "incremental_mRMR_score":max_score}
    if verbose:  
        print(f"mRMR incremental for {max_features} features took {round(time.time()-t0,4)} seconds")
    # # compute mRMR of each feature subset
    # for S_, d_ in outputs.items():
    #     rel_ = np.mean(relevances[S_])
    #     red_ = 
    #     for i in S_:
    #         for j in S_:


    return outputs, relevances, pairwise_redundancies

def perform_lasso(input_data_post_mrmr:pd.DataFrame, targets:pd.DataFrame, N_CPUS=1, method="cv", verbose=False):
    """
    Performs Lasso regression on the data.
    Arguments:
    ----------
    input_data_post_mrmr: pd.DataFrame, input data after performing minimum redundancy maximum relevance feature selection
    targets: pd.DataFrame, target values
    N_CPUS: int, number of cpus available for parallel computation, 
            -1: use all cpus
            None: do not parallelize
    method: str, choices=["cv","aic"]
            method for selecting the best alpha value for Lasso regression
            cv: cross-validation
            aic: Akaike Information Criterion using LassoLarsIC
    
    Returns:
    --------
    """
    if N_CPUS is not None:
        raise NotImplementedError("parallelization not implemented yet")

    vs_ssvr = VS_SSVR()

    new_alpha_values_exp = np.linspace(-4,3,200) # 200 alpha values to test between 1e-4 and 1e3
    new_alpha_values = 10**new_alpha_values_exp

    vs_ssvr.alpha_values = new_alpha_values

    t0 = time.time()
    if method=="cv":
        results = vs_ssvr.variable_selection_sparse_svr(
            input_data_post_mrmr, targets, n_splits=10, max_iter=1000
        )
        if verbose:
            print("time to perform lasso cv for each alpha values:",time.time()-t0)
    elif method=="aic":
        # use LassoLarsIC
        scaler = StandardScaler()
        input_data_post_mrmr = scaler.fit(input_data_post_mrmr)
        _scaled_data = scaler.transform(input_data_post_mrmr)
        _lasso_lars_ic = LassoLarsIC(criterion="aic")
        _lasso_lars_ic.fit(_scaled_data, targets)
        results = _lasso_lars_ic 
        return results
    else:
        raise ValueError("method arg must be one of ['cv','aic']")


    n_splits = len(results)-1 # -1 because of 'indices_above_dummy_threshold' key

    important_features = results['split_0']['coefficients']
    for i in range(1,n_splits):
        important_features += results[f'split_{i}']['coefficients']
    important_features /= n_splits

    # set those <= dummy threshold to zero
    indices_to_zero = set(list(range(len(important_features)))) - set(results['indices_above_dummy_threshold'])
    for i in indices_to_zero:
        important_features[i]=0.0

    if ( important_features[-10:]==np.zeros(10) ).all():
        print("all dummy variable coefficients zeroed out")

    feature_weight_df = pd.DataFrame({
        "feature_name":list(input_data_post_mrmr.columns),
        "feature_idx" :list(range(len(input_data_post_mrmr.columns))),
        "weight":important_features,
        "abs_weight":abs(important_features)
    })
    feature_weight_nonzero_df = feature_weight_df[feature_weight_df['abs_weight']>0]
    feature_weight_nonzero_df.head()

    sorted_features_df = feature_weight_nonzero_df.sort_values("abs_weight", ascending=False, ignore_index=True)
    if verbose:
        print("top 15 features:")
        print(sorted_features_df.head(15))

    return results, sorted_features_df

class VS_SSVR:
    def __init__(self, 
                 n_splits:int=10, 
                 n_dummy_features:int=10, 
                 C_values:list[float]=[0.1, 1.0, 10.0], 
                 epsilon_values:list[float]=[0.1, 0.5, 1.0],
                 feature_selection_model = "lasso"
        ):
        """
        Object that performs Variable Selection using a Sparse Linear SVR model. 

        Arguments:
        ----------
        n_splits: int, number of splits for variable selection. 
                    different data splits may yield different selected features, 
                    we try to mitigate that by averaging coefficients. 
        
        n_dummy_features: int, number of dummy features to add to the data.
                            used for finding a threshold of 'variable importance'. 
        
                            
        Notes:
        ------

        Based on:
        - Bi et al. (2003) "Dimensionality Reduction via Sparse Support Vector Machines"
        and applied to the problem of predicting antimicrobial activity of peptides in:
        - Lee et al. (2016) "Mapping membrane activity in undiscovered peptide sequence space using machine learning"


        """

        self.n_splits = n_splits
        self.n_dummy_features = n_dummy_features
        self.set_hyperparameters(C_values, epsilon_values)
        
        self.feature_selection_model = feature_selection_model

        self.coefficient_threshold = None
        self.alpha_values = [1e-4,1e-3,1e-2, 1e-1, 0.5, 1.0, 1.5, 2.5, 5.0, 7.5, 10.0, 50, 10**2]
        self.results = None

    def set_hyperparameters(self, C_values:list[float], epsilon_values:list[float]):
        """
        Sets the hyperparameters for the SVR model selection. 

        Arguments:
        ----------
        C_values: list[float], list of C values to try
        epsilon_values: list[float], list of epsilon values to try
        """
        self.C_values = C_values
        self.epsilon_values = epsilon_values
        print("Hyperparameters (re)set. Must re-reun variable selection.")

    def get_hyperparameters(self):
        """
        Returns the hyperparameters for the SVR model selection. 

        Returns:
        --------
        C_values: list[float], list of C values to try
        epsilon_values: list[float], list of epsilon values to try
        """
        return self.C_values, self.epsilon_values

    def get_threshold(self):
        """
        Returns the threshold for variable selection. 

        Returns:
        --------
        threshold: float, threshold for variable selection
        """
        return self.coefficient_threshold

    def get_results(self):
        """
        Returns the results of the variable selection. 

        Returns:
        --------
        results: dict, dictionary containing the following keys:
            "split_i" : dict, dictionary containing the following
                "non_zero_indices" : np.array, indices of non-zero coefficients
                "coefs" : np.array, coefficients of the model
                "mse_on_test" : float, mean squared error on test set
            "common_non_zero_indices" : list, common non-zero indices across all splits
            "indices_above_dummy_threshold" : list, indices of features that are above the dummy threshold
        """
        return self.results

    def model_selection_lasso(self, train_X, train_Y, test_X, test_Y, max_iter=1000):
        """
        Performs model selection using a grid search over alpha values, hyperparameter 
        controlling the L1-regularization in Lasso regression.
        Arguments:
        ----------
        X: pd.DataFrame, input data
        y: pd.DataFrame, target values

        Returns:
        --------
        results: dict, dictionary containing the following keys:
            "best_params": dict, best parameters (C, epsilon) found
            "best_mse": float, mean squared error on test set of best parameters
            "coefficients": np.array, coefficients of the best model
        """
        alpha_values = self.alpha_values
        results = {}
        best_mse = np.inf
        best_params = {"alpha":None}
        best_coefficients = None
        for alpha_ in alpha_values:
            lasso_regression = Lasso(alpha=alpha_,max_iter=max_iter)

            # fit model
            lasso_regression.fit(train_X, train_Y)

            # get mse of model
            y_pred_ = lasso_regression.predict(test_X)
            mse_ = mean_squared_error(test_Y, y_pred_)

            # only keep best parameters
            if mse_ < best_mse:
                best_mse = mse_
                best_params["alpha"] = alpha_
                best_coefficients = lasso_regression.coef_

        
        results[f"best_params"] = best_params
        results[f"best_mse"] = best_mse
        results[f"coefficients"] = best_coefficients
        return results

    def model_selection(self, train_X, train_Y, test_X, test_Y):
        """
        Performs model selection using a grid search over C and epsilon values (object attributes).

        Arguments:
        ----------
        X: pd.DataFrame, input data
        y: pd.DataFrame, target values

        Returns:
        --------
        results: dict, dictionary containing the following keys:
            "best_params": dict, best parameters (C, epsilon) found
            "best_mse": float, mean squared error on test set of best parameters
            "coefficients": np.array, coefficients of the best model
        """
        C_values = self.C_values
        epsilon_values = self.epsilon_values
        results = {}
        best_mse = np.inf
        best_params = {"C":None, "epsilon":None}
        for C_ in C_values:
            for epsilon_ in epsilon_values:
                sparse_linear_svr = LinearSVR(
                    loss="epsilon_insensitive", # epsilon_insensitive is basically L1 loss
                    C=C_, 
                    epsilon=epsilon_
                )

                # fit model
                sparse_linear_svr.fit(train_X, train_Y)

                # get mse of model
                y_pred_ = sparse_linear_svr.predict(test_X)
                mse_ = mean_squared_error(test_Y, y_pred_)

                # only keep best parameters
                if mse_ < best_mse:
                    best_mse = mse_
                    best_params["C"] = C_
                    best_params["epsilon"] = epsilon_

        
        results[f"best_params"] = best_params
        results[f"best_mse"] = best_mse
        results[f"coefficients"] = sparse_linear_svr.coef_
        return results
        
    def variable_selection_sparse_svr(self, data:pd.DataFrame, targets:pd.DataFrame, n_splits:int=5, n_dummy_features:int=10, max_iter:int=1000):
        """
        Performs variable selection using a sparse linear SVR model.
        Based on 
        - Lee et al. (2016) "Mapping membrane activity in undiscovered peptide sequence space using machine learning"
        - Bi et al. (2003) "Dimensionality Reduction via Sparse Support Vector Machines"

        Arguments:
        ----------
        data    : pd.DataFrame, input data. This will be split for cross-validation.
        targets : pd.DataFrame, target values for regression.
        n_splits: int, number of splits for cross-validation.
        n_dummy_features: int, number of dummy features to add to the data. 
                            If a variable has a coefficient less than the dummy features, 
                            then it should be removed
        max_iter: int, maximum number of iterations for optimizing LASSO model
        
        Returns:
        --------
        results : dict, dictionary containing the following keys:
            "split_i" : dict, dictionary containing the following
                "non_zero_indices" : np.array, indices of non-zero coefficients
                "coefs" : np.array, coefficients of the model
                "mse_on_test" : float, mean squared error on test set
            "common_non_zero_indices" : list, common non-zero indices across all splits
            "indices_above_dummy_threshold" : list, indices of features that are above the dummy threshold
        """
        
        # shuffler = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        shuffler = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        C_values = [0.1, 1.0, 10.0]
        epsilon_values = [0.1, 0.5, 1.0]

        # add dummy features
        for i in range(n_dummy_features):
            data[f"dummy_{i}"] = np.random.normal(0, 1, data.shape[0])    

        scaler = StandardScaler()
        ############################
        # for each split, perform grid search and collect results
        results = {}
        for i, (train_index, test_index) in enumerate(shuffler.split(data)):
            train_X_ = data.iloc[   train_index]
            train_Y_ = targets.iloc[train_index]

            test_X_  = data.iloc[   test_index]
            test_Y_  = targets.iloc[test_index]

            # standardize data
            scaler.fit(train_X_)
            train_X_ = scaler.transform(train_X_)
            test_X_  = scaler.transform( test_X_)

            ############################
            # perform grid search for best parameters given the split
            # must do this b/c the coefs may change depending on the hyperparams. 
            # collects best_params, best_mse, and coefficients
            if self.feature_selection_model == "lasso":
                t0_ = time.time()
                results[f"split_{i}"] = self.model_selection_lasso(
                    train_X_, train_Y_, test_X_, test_Y_, max_iter=max_iter
                )
                tf_ = time.time()
            else:
                t0_ = time.time()
                results[f"split_{i}"] = self.model_selection(
                    train_X_, train_Y_, test_X_, test_Y_
                )
                tf_ = time.time()
            logging.info(f"Split {i+1}/{n_splits} took {round(tf_ - t0_,4)} seconds")

        # find common non-zero indices
        # NOT DONE YET

        # get average of all coefficients
        avg_coefs = np.zeros(data.shape[1])
        for split_ in results:
            avg_coefs += np.array(results[split_]["coefficients"])
        avg_coefs /= n_splits

        # take average of dummy variable coefficients as threshold 
        dummy_variable_coefs = []
        for split_ in results:
            dummy_variable_coefs += list(np.array(results[split_]["coefficients"])[-n_dummy_features:])
        threshold = np.mean( np.abs(dummy_variable_coefs) )
        self.coefficient_threshold = threshold
        
        # get indices of features that are above threshold
        selected_indices = np.where(
            np.abs(avg_coefs[:len(avg_coefs)-n_dummy_features]) > threshold
        )[0]
        results["indices_above_dummy_threshold"] = selected_indices
        self.results = results
        return results


class NonlinearSVRonPhysicoChemicalProps():
    def __init__(self, target_name:str, important_feature_indices:list[int], kernel:str="rbf", C:float=1.0, epsilon:float=0.1):
        self.target_name = target_name # what is this model predicting, mic, log_mic, etc
        self._selected_features = important_feature_indices
        self.svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
        self.scaler = StandardScaler()

    def set_model_params(self, model_params:dict):
        """
        Sets the parameters for the SVR model, scaler, and target name. 
        For re-loading a model.

        Arguments:
        ----------
        model_params: dict, dictionary containing 
            "svr_params": dict, parameters for the SVR model
            "scaler_params": dict, parameters for the StandardScaler
            "target_name": str, name of the target variable
        """
        svr_params    = model_params["svr_params"]
        scaler_params = model_params["scaler_params"]
        target_name   = model_params["target_name"]
        
        self.svr.set_params(**svr_params)
        self.scaler.set_params(**scaler_params)
        self.target_name = target_name
    
    def get_model_params(self):
        """
        Returns the dict of parameters for the SVR model, scaler, and target name. 

        Returns:
        --------
        outputs: dict, dictionary containing
            "svr_params": dict, parameters for the SVR model
            "scaler_params": dict, parameters for the StandardScaler
            "target_name": str, name of the target variable
        """
        outputs = {}
        outputs[    "svr_params"] = self.svr.get_params()
        outputs[ "scaler_params"] = self.scaler.get_params()
        outputs[   "target_name"] = self.target_name
        return outputs

    def get_physico_chemical_props(self,sequences:list[str]|str)->pd.DataFrame:
        """
        uses compute_propy_properties to get physicochemical properties of sequences
        then filters out the selected features
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        physicochemical_props_ = compute_propy_properties(sequences)
        # check if selected features contains indices or column names
        if all([isinstance(f_,int) for f_ in self._selected_features]):
            physicochemical_props_ = physicochemical_props_.iloc[:,self._selected_features]
        else:
            try:
                physicochemical_props_ = physicochemical_props_[self._selected_features]
            except KeyError:
                return None

        return physicochemical_props_

    def fit(self, X, y, input_type="sequences"):
        """
        Arguments:
        ----------
            X: array, input values
            y: array, target values
            input_type: str, choices=["sequences","physicochemical_properties"]
        """
        if input_type not in ["sequences", "physicochemical_properties"]:
            raise ValueError("input_type must be one of ['sequences','physicochemical_properties']")
        
        if input_type=="sequences":
            physicochemical_props_ = self.get_physico_chemical_props(X)
        else:
            physicochemical_props_ = X
        
        # standardize data
        self.scaler.fit(physicochemical_props_)
        physicochemical_props_ = self.scaler.transform(physicochemical_props_)

        # fit svr model
        self.svr.fit(physicochemical_props_, y)
    
    def predict(self, sequences:list[str]|str):
        """
        wrapper predict method to automatically get physicochemical properties given (a) new sequence(s)
        then predict from that

        Arguments:
        ----------
        sequences: str or list[str], sequences to predict off of

        Returns:
        --------
        predicted values, 
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # compute sequence features/physicochemical properties
        physicochemical_props_ = self.get_physico_chemical_props(sequences)
        if physicochemical_props_ is None:
            return None
        
        # put through scaler
        physicochemical_props_ = self.scaler.transform(physicochemical_props_)

        # predict
        y_predicted = self.svr.predict(physicochemical_props_)
        return y_predicted

class SVRonPhysicoChemicalProps():
    def __init__(self, svr_params, important_features, svr_C=1.0, svr_epsilon=0.1):
        

        self.svr_C = svr_C
        self.svr_epsilon = svr_epsilon
        self.important_features = important_features
        self.important_feature_cols = None
        
        self.scaler = StandardScaler()
        self.svr = SVR()
        self.svr.set_params(**svr_params)

    def get_physico_chemical_props(self,sequences:list[str]|str):

        if isinstance(sequences, str):
            sequences = [sequences]

        titles=[
            'Aliphatic Index', 'Boman Index', 
            'Charge pH 3', 'Charge pH 7', 'Charge pH 11', 
            'Hydrophobicity', 'Instability Index', 
            'Isoelectric Point', 'Molecular Weight'
        ]
        physico_props = {
            "sequence":[]
        }
        for title_ in titles:
            physico_props[title_] = []
            
        for seq_ in sequences:
            peptide_ = peptides.Peptide(seq_)
            physico_props["sequence"].append(seq_)
            for title_ in titles:
                if title_ == "Aliphatic Index":
                    physico_props[title_].append(peptide_.aliphatic_index())
                elif title_ == "Boman Index":
                    physico_props[title_].append(peptide_.boman())
                elif title_ == "Charge pH 3":
                    physico_props[title_].append(peptide_.charge(pH=3))
                elif title_ == 'Charge pH 7':
                    physico_props[title_].append(peptide_.charge(pH=7))
                elif title_ == "Charge pH 11":
                    physico_props[title_].append(peptide_.charge(11))
                elif title_ == "Hydrophobicity":
                    physico_props[title_].append(peptide_.hydrophobicity())
                elif title_ == "Instability Index":
                    physico_props[title_].append(peptide_.instability_index())
                elif title_ == "Isoelectric Point":
                    physico_props[title_].append(peptide_.isoelectric_point())
                elif title_ == "Molecular Weight":
                    physico_props[title_].append(peptide_.molecular_weight())
        
        output = pd.DataFrame(data=physico_props)
        return output[ output.columns[1:] ]

    def get_propy_props(self, sequences:list[str]|str):
        
        if isinstance(sequences, str):
            sequences = [sequences]

        min_len = 2 # from grampa dataset
        lambda_max = max(4, min_len-1)
        results = {}
        for i, seq in enumerate(sequences):
            
            succeeded = False
            Des = GetProDes(seq)
            
            if len(seq)<5:
                succeeded = False
            else:
                try:
                    alldes_ = Des.GetALL(
                        paac_lamda =lambda_max,
                        apaac_lamda=lambda_max,
                    )
                    succeeded = True
                except:
                    print(f"failed with seq={seq}, {len(seq)=}")
                    succeeded = False

            if succeeded:
                if i==0:
                    for key_ in alldes_:
                        results[key_] = [alldes_[key_]]
                else:
                    for key_ in alldes_:
                        results[key_].append(alldes_[key_])
            else:
                # if propy was not able to compute properties, fill with np.nan
                for key_ in results:
                    if key_=="sequence":
                        continue
                    else:
                        results[key_].append(np.nan)
        results = pd.DataFrame(results)
        results = results[results.columns[1:]]

        if self.important_feature_cols is not None:
            results = results[self.important_feature_cols]
        else:
            important_features = list(self.important_features)
            results = results.iloc[:,important_features]
        return results

    def fit(self, X, y, input_type="sequences", pcprops="propy"):
        """
        Arguments:
        ----------
            X: array, input values
            y: array, target values
            input_type: str, choices=["sequences","physicochemical_properties"]
        """
        if input_type not in ["sequences", "physicochemical_properties"]:
            raise ValueError("input_type must be one of ['sequences','physicochemical_properties']")
        
        if input_type=="sequences":
            if pcprops=="propy":
                physicochemical_props_ = self.get_propy_props(X)
            else:
                physicochemical_props_ = self.get_physico_chemical_props(X)
        else:
            physicochemical_props_ = X
        
        # standardize data
        self.scaler.fit(physicochemical_props_)
        physicochemical_props_ = self.scaler.transform(physicochemical_props_)

        # fit svr model
        self.svr.fit(physicochemical_props_, y)

    def predict(self, sequences:list[str]|str, pcprops="propy"):
        """
        wrapper predict method to automatically get physicochemical properties given (a) new sequence(s)

        Arguments:
        ----------
        sequences: str or list[str], sequences to predict off of

        Returns:
        --------
        predicted values, 
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if pcprops=="propy":
            physicochemical_props_ = self.get_propy_props(sequences)
        else:
            physicochemical_props_ = self.get_physico_chemical_props(sequences)

        # put through scaler
        physicochemical_props_ = self.scaler.transform(physicochemical_props_)
        
        y_predicted = self.svr.predict(physicochemical_props_)
        return y_predicted