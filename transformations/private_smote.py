"""This file applies PrivateSMOTE to original data by producing several data variants.
"""
# %%
# pylint: disable=wrong-import-position

from os import sep
from collections import defaultdict
from random import randrange
import random
import ast
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

sys.path.append('../')
from dataprep import preprocessing

# %%
def encode(data) -> pd.DataFrame:
    """transform string numbers to numeric type

    Args:
        data (pd.Dataframe): data

    Returns:
        pd.DataFrame: encoded data
    """
    for col in data.columns:
        try:
            data[col] = data[col].apply(lambda x: ast.literal_eval(x))
        except Exception:
            # not a string
            pass

    return data


def aux_singleouts(key_vars, data) -> pd.DataFrame:
    """find single outs according k-anonymity measure for a set of quasi-identifiers

    Args:
        key_vars (str): set of quasi-identifiers
        data (pd.Dataframe): data

    Returns:
        pd.Dataframe: data with single outs
    """
    k = data.groupby(key_vars)[key_vars[0]].transform(len)
    # data['single_out'] = None
    data['single_out'] = np.where(k == 1, 1, 0)
    return data


class Smote:
    """Apply Smote
    """
    def __init__(self, samples, y, N, k):
        """Initiate arguments

        Args:
            samples (array): training samples
            y (1D array): target sample
            N (int): number of interpolations per observation
            k (int): number of nearest neighbours
        """
        self.n_samples = samples.shape[0]
        self.n_attrs = samples.shape[1]
        self.y = y
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs+1))

    def over_sampling(self):
        """find the nearest neighbors and populate with new data

        Returns:
            pd.DataFrame: synthetic data
        """
        N = int(self.N)

        neighbors = NearestNeighbors(n_neighbors=self.k+1).fit(self.samples)

        # for each observation find nearest neighbours
        for i, _ in enumerate(self.samples):
            nnarray = neighbors.kneighbors(
                self.samples[i].reshape(1, -1), return_distance=False)[0]
            self._populate(N, i, nnarray)

        return self.synthetic

    def _populate(self, N, i, nnarray):
        # populate N times
        for j in range(N):
            # find index of nearest neighbour excluding the observation in comparison
            neighbour = randrange(1, self.k+1)

            difference = self.samples[nnarray[neighbour]] - self.samples[i]
            # multiply with a weight
            weight = random.uniform(0, 1)
            additive = np.multiply(difference, weight)

            # assign interpolated values
            self.synthetic[self.newindex, 0:len(
                self.synthetic[self.newindex])-1] = self.samples[i]+additive
            # assign intact target variable
            self.synthetic[self.newindex, len(
                self.synthetic[self.newindex])-1] = self.y[i]
            self.newindex += 1

# %% privateSMOTE regardless of the class


def private_smote():
    """Apply PrivateSMOTE to each original data
    """
    output_folder = 'output/data_variants/PrivateSMOTE/'
    key_vars = preprocessing.quasi_identifiers()
    all_data = preprocessing.read_data()
    indexes = preprocessing.get_indexes()
    for i, _ in enumerate(all_data["data"]):
        index = indexes[i]
        data = all_data["data"][i]
        # get 80% of data to synthesise
        data_idx = list(set(list(data.index)) - set(index))
        data = data.iloc[data_idx, :]

        # encode string with numbers to numeric
        data = encode(data)
        # apply LabelEncoder to categorical attributes
        label_encoder_dict = defaultdict(LabelEncoder)
        data_encoded = data.apply(lambda x: label_encoder_dict[x.name].fit_transform(
            x) if x.dtype == 'object' else x)
        # remove trailing zeros in integers
        data_encoded = data_encoded.apply(
            lambda x: x.astype(int) if all(x % 1 == 0) else x)

        map_dict = dict()
        for k in data.columns:
            if data[k].dtype == 'object':
                keys = data[k]
                values = data_encoded[k]
                sub_dict = dict(zip(keys, values))
                map_dict[k] = sub_dict

        qis = key_vars[i]
        data_wout_singleouts = aux_singleouts(qis, data_encoded)
        X_train = data_wout_singleouts.loc[data_wout_singleouts['single_out']
                                        == 1, data_wout_singleouts.columns[:-2]]
        Y_train = data_wout_singleouts.loc[data_wout_singleouts['single_out']
                                        == 1, data_wout_singleouts.columns[-1]]
        y = data_wout_singleouts.loc[data_wout_singleouts['single_out']
                                    == 1, data_wout_singleouts.columns[-2]]

        # getting the number of singleouts in training set
        singleouts = Y_train.shape[0]

        # storing the singleouts instances separately
        x1 = np.ones((singleouts, X_train.shape[1]))
        x1 = [X_train.iloc[i] for i, v in enumerate(Y_train) if v == 1.0]
        x1 = np.array(x1)

        y = np.array(y)

        knn = [1, 3, 5]
        per = [1, 2, 3]
        for k in knn:
            for p in per:
                new = Smote(x1, y, p, k).over_sampling()
                newDf = pd.DataFrame(new)
                # restore feature name
                newDf.columns = data_wout_singleouts.columns[:-1]
                # assign singleout
                newDf[data_wout_singleouts.columns[-1]] = 1
                # add non single outs
                newDf = pd.concat(
                    [newDf, data_wout_singleouts.loc[data_wout_singleouts['single_out'] == 0]])
                for col in newDf.columns:
                    if data_wout_singleouts[col].dtype == np.int64:
                        newDf[col] = round(newDf[col], 0).astype(int)
                    elif data_wout_singleouts[col].dtype == np.float64:
                        # get decimal places in float
                        dec = str(data_wout_singleouts[col].values[0])[
                            ::-1].find('.')
                        newDf[col] = round(newDf[col], dec)
                    else:
                        newDf[col] = newDf[col].astype(
                            data_wout_singleouts[col].dtype)

                # decoded
                for key in map_dict.keys():
                    dict_items = dict(map(reversed, map_dict[key].items()))
                    newDf[key] = newDf[key].apply(lambda x: dict_items.get(
                        x) or dict_items[min(dict_items.keys(), key=lambda key: abs(key-x))])

                # save oversampled data
                newDf.to_csv(
                    f'{output_folder}{sep}{all_data["name"][i]}_knn{k}_per{p}.csv',
                    index=False)


# %%


private_smote()
