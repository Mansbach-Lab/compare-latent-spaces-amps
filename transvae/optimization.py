import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


import pandas as pd
import numpy as np
import torch
import os
import pickle as pkl
import logging

from transvae import trans_models
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN
from transvae.tvae_util import *
from scripts.parsers import model_init, train_parser

from sklearn.decomposition import PCA
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, LogExpectedImprovement 
from botorch.optim import optimize_acqf

# from ..oracles.AMPlify.src.AMPlify import Amplify

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


class OptimizeInReducedLatentSpace():
    def __init__(self, 
                 generative_model, 
                 property_oracle, 
                 dimensionality_reduction,
                 char_dict,
                 minimize_or_maximize_score="maximize",
                 params={}
        ):
        """
        Class to perform Bayesian Optimization in the reduced latent space of a generative model.

        Parameters
        ----------
        generative_model : torch.nn.Module-like
            A trained generative model that can be used to generate sequences.
            Should have a method `greedy_decode` that can decode a latent space vector to a sequence.
        property_oracle :  
            Supervised learning model that predicts a property of a sequence.
            Must have a method `predict` that takes a list of sequences and 
            returns a list of prediction scores (probability, regression value, etc).
        dimensionality_reduction : 
            A dimensionality reduction model. Similar to sklearn's PCA.
            Must have a method `transform` and `inverse_transform`.
        char_dict : dict
            A dictionary that maps sequence characters to integers.
        minimize_or_maximize_score : str, optional
            Whether to minimize or maximize the property score. 
            If "maximize", the optimizer will attempt to maximize the score.
            If "minimize", the optimizer will attempt to minimize the score.
            The default is "maximize".
        params : dict, optional
            Additional options, mainly to check if using ESM or other generative models.
        """
        
        # Generative Model and Property Oracle setup
        self.generative_model = generative_model
        self.char_dict = char_dict
        self.minimize_or_maximize_score = minimize_or_maximize_score
        self.property_oracle = property_oracle
        self.params = params

        print("Testing property oracle...")
        test_sequence = "MKKLLLLLLLLLLRRLAAASLKLSN"
        test_prediction = self.property_oracle.predict([test_sequence])
        if isinstance(test_prediction, tuple):
            raise ValueError("""Property Oracle must return a single value, not a tuple.
                             Try wrapping it in a class that only returns the prediction value.""")

        # Dimensionality Reduction setup
        # self.n_components = n_dim_components
        self.dimensionality_reducer = dimensionality_reduction
        self.n_reduced_dims = self.dimensionality_reducer.n_components

        # bounds for Upper Confidence Bound method of 'exploring' the latent space
        self.bounds = torch.stack([torch.ones(self.n_reduced_dims)*(-10), torch.ones(self.n_reduced_dims)*10])

        self.optimization_results = {
            "iterations":[],
            "candidates":[],
            "candidate_scores":[],
            "best_objective_values":[],
            "best_sequences":[],
            'params':params
        }

    def decode_seq(self, encoded_seq:list[int]) -> str:
        """
        Decodes an encoded sequence to a list of characters.

        Parameters
        ----------
        encoded_seq : list[int]
            A list of integers representing a sequence.

        Returns
        -------
        str
            The string representation of the sequence.
        """
        itos = {v:k for k,v in self.char_dict.items()}
        output = "".join([itos[i] for i in encoded_seq])
        output = output.strip("_")
        output = output.strip("<start>")
        output = output.strip("<end>")
        return output

    def encode_seq(self, sequence:str) -> list[int]:
        """
        Encodes a sequence to a list of integers.

        Parameters
        ----------
        sequence : str
            A sequence of characters.

        Returns
        -------
        list[int]
            A list of integers representing the sequence.
        """
        stoi = self.char_dict
        output = [stoi[i] for i in sequence]
        return output
        
    def get_fitted_gp_model(self, train_X, train_Y):
        """
        Fits the Gaussian Process model to the given data.

        Parameters
        ----------
        train_X : torch.Tensor
            The input data.
        train_Y : torch.Tensor
            The output data.
        """
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def optimize(self, train_X, train_Y, n_iters=10, n_restarts=5,verbose=False):

        train_X=train_X.to(float)
        train_Y=train_Y.to(float)

        char_dict = self.char_dict
                
        if self.optimization_results["iterations"]:
            last_iteration = self.optimization_results["iterations"][-1]
            best_score = self.optimization_results["best_objective_values"][-1]
            best_seq = self.optimization_results["best_sequences"][-1]
        else:
            print("============================================\nStarting optimization...")
            last_iteration = 0

            # get best from provided initialization data
            if self.minimize_or_maximize_score=="minimize":
                _idx = train_Y.argmin()
                best_score = train_Y.min().item()
            else:
                _idx = train_Y.argmax() 
                best_score = train_Y.max().item()

            _inv_proj = self.dimensionality_reducer.inverse_transform(
                train_X[[_idx]]
            )
            

            if isinstance(_inv_proj, np.ndarray):
                _inv_proj = torch.from_numpy(_inv_proj)
            
            _inv_proj = _inv_proj.to(torch.float32)
            
            if _inv_proj.dim() == 1:
                _inv_proj.unsqueeze_(0)

            with torch.no_grad():
                if self.params.get("use_esm", False):
                    _decode_inv_proj = self.generative_model.greedy_decode(_inv_proj)
                    best_seq = _decode_inv_proj # ESM has its own decoding method
                else:
                    _decode_inv_proj = self.generative_model.greedy_decode(_inv_proj);
                    best_seq = self.decode_seq(_decode_inv_proj.flatten().numpy())

        for i in range(n_iters):

            # fit and get the GP model
            self.gp = self.get_fitted_gp_model(train_X, train_Y)
            # self.UCB = UpperConfidenceBound(self.gp, beta=0.1)
            # self.acq_func = ExpectedImprovement(self.gp, best_score)
            self.acq_func = LogExpectedImprovement(self.gp, best_score)

            # grab candidate(s)
            # q x d if return_best_only is True (default) – num_restarts x q x d if return_best_only is False
            candidates, acq_values = optimize_acqf(
                self.acq_func, bounds=self.bounds, q=1, num_restarts=n_restarts, raw_samples=20,
                return_best_only=False
            )
            
            # inverse projection
            candidates_invproj = self.dimensionality_reducer.inverse_transform(candidates)
            if verbose:
                print(f"candidate in reduced space: {candidates}")
                print(f"candidate in original space: {candidates_invproj}")

            if isinstance(candidates_invproj, np.ndarray):
                candidates_invproj = torch.from_numpy(candidates_invproj)

            candidates_invproj = candidates_invproj.reshape(-1, candidates_invproj.shape[-1]) # (n_restarts, 1, d_latent) -> (n_restarts, d_latent)
            # decode the candidate sequence from the latent space to the original sequence space
            with torch.no_grad():
                if self.params.get("use_esm", False):
                    _decode_inv_proj = self.generative_model.greedy_decode(candidates_invproj)
                    candidate_decoded = _decode_inv_proj # ESM has its own decoding method
                else:
                    candidate_decoded = self.generative_model.greedy_decode(candidates_invproj);
            
            if self.params.get("use_esm", False):
                candidate_sequences = candidate_decoded
            else:
                candidate_sequences = []
                for j in range(candidate_decoded.shape[0]):
                    candidate_sequences.append(self.decode_seq(candidate_decoded[j].flatten().numpy()))

            if verbose:
                print(f"candiate sequences: {candidate_sequences}")
            
            prediction_scores = self.property_oracle.predict(
                candidate_sequences
            )
            
            candidate          = candidates[         acq_values.argmax()]
            candidate_sequence = candidate_sequences[acq_values.argmax()]

            if prediction_scores is None: 
                prediction_score = None
            else:
                prediction_score   = prediction_scores[  acq_values.argmax()]

            # if all([i is None for i in prediction_scores]):
            #     prediction_score = None

            if   isinstance(prediction_score, torch.Tensor):
                prediction_score = prediction_scores.item()
            elif isinstance(prediction_score, np.ndarray):
                prediction_score = prediction_scores[0]
            elif isinstance(prediction_score, list):
                prediction_score = prediction_scores[0]

            if self.minimize_or_maximize_score=="minimize":
                if prediction_score is None:
                    prediction_score = 3.5 # for MIC and HC50, the log values range between, -1.5 to 3.5. so 3.5 is a reasonable upper bound
                objective_value = -prediction_score
            else:
                if prediction_score is None:
                    prediction_score = -3.5
                objective_value =  prediction_score

            # update the training data for the GP model
            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat([train_Y, torch.tensor(prediction_score).float().reshape(1,1)], dim=0)
            if objective_value > best_score:
                best_score = objective_value
                best_seq   = candidate_sequence
            


            print(f"iteration {i+1} completed. Prediction score: {prediction_score}")
            self.optimization_results["iterations"].append(i+1+last_iteration)
            self.optimization_results["candidates"].append(candidate_sequence)
            self.optimization_results["candidate_scores"].append(prediction_score)
            self.optimization_results["best_objective_values"].append(best_score)
            self.optimization_results["best_sequences"].append(best_seq)

            if i%100==0:
                _run = self.params.get("run", "default")
                _name = self.params.get("chkpt_fpath", "default").split('/')[1]
                _output_dir = self.params.get("output_dir", "./")
                if _output_dir[-1]!="/":
                    _output_dir+="/"
                with open(f"{_output_dir}optimization_results_{_name}_run{_run}.pkl", "wb") as f:
                    pkl.dump(self.optimization_results, f)

        print("Optimization complete")


if __name__ == "__main__":

    #########################################
    # load some training data
    print("loading data...")
    data_fpath = "data/"
    train_seqs = pd.read_csv(data_fpath+"peptides_2024_train.txt")
    train_fctn = pd.read_csv(data_fpath+"peptides_2024_train_function.txt")
    with open(data_fpath+"char_dict_peptides_2024.pkl", 'rb') as f:
        char_dict = pkl.load(f)

    df = encode_seqs(train_seqs, char_dict)    
    df["amp_or_not"] = train_fctn['amp']

    n_amps     = 100
    n_non_amps = 100
    some_amps     = df[df.amp_or_not==1].sample(    n_amps)
    some_non_amps = df[df.amp_or_not==0].sample(n_non_amps)
    
    sampled_peptides = []
    sampled_fctns = []
    for i,seq in enumerate(some_amps["encoded_peptides"]):
        sampled_peptides.append( [decode_seq(seq,char_dict)] )
        sampled_fctns.append( [some_amps.iloc[i,1]])
    for i,seq in enumerate(some_non_amps["encoded_peptides"]):
        sampled_peptides.append( [decode_seq(seq,char_dict)] )
        sampled_fctns.append( [some_non_amps.iloc[i,1]])

    sampled_peptides = np.array(sampled_peptides)
    sampled_fctns = np.array(sampled_fctns)


    #########################################
    # load a trained generative model
    print("loading generative model...")
    model_src = "checkpointz/amp_rnn_organized/070_rnn-128_peptides_2024.ckpt"
    model_obj=torch.load(model_src, map_location=torch.device("cpu"))
    # model = TransVAE(load_fn=model_src, workaround="cpu")
    model = RNN(load_fn=model_src, workaround="cpu")
    model.params['HARDWARE']= 'cpu'

    model.params["BATCH_SIZE"] = n_amps + n_non_amps
    t0 = time.time()
    with torch.no_grad():
        z, mu, logvar = model.calc_mems(sampled_peptides, log=False,save=False)
        decoded_seqs  = model.reconstruct(np.c_[sampled_peptides,sampled_fctns],log=False,return_mems=False)
    print(f"time elapsed = {round(time.time()-t0,5)}s")

    #########################################
    # build a dimensionality reduction method
    print("building PCA...")
    pca = PCA(n_components=5)
    pca.fit(mu)
    pca_mu = pca.transform(mu)

    #########################################
    # load a trained property predictor/oracle
    # print("loading amplify...")
    # amplify = Amplify("oracles/AMPlify/models/", "balanced")

    #########################################
    # initialize the optimizer
    print("initializing optimizer...")
    train_X = torch.from_numpy(pca_mu.copy())
    train_Y = pd.concat([some_amps, some_non_amps], axis=0, ignore_index=True)
    train_Y = torch.from_numpy(train_Y[["amp_or_not"]].values)

    # optimizer = OptimizeInReducedLatentSpace(
    #     model, amplify, pca, char_dict
    # )

    # #########################################
    # # perform optimization
    # print("optimizing...")
    # optimizer.optimize(train_X, train_Y, n_iters=10)