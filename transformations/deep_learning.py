"""This script applies deep learning strategies for data synthetisation
"""
# %%
# pylint: disable=wrong-import-position
from os import sep
import sys
import pandas as pd
import numpy as np
from sdv.tabular import CTGAN, TVAE, CopulaGAN
sys.path.append('./')
from dataprep.preprocessing import get_indexes, quasi_identifiers, read_data


# %%
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

# %%

def aplly_deep_learning(msg):
    """Apply deep learning strategies for data synthetisation

    Args:
        msg (str): name of the original file and respective GAN technique with parameters
    """
    output_folder = 'output/data_variants/deep_learning_conditional_sampling/'

    key_vars = quasi_identifiers()
    all_data = read_data()
    # get 80% of data to synthesise
    indexes = get_indexes()
    file = msg.split('_')[0]
    names_index = all_data['name'].index(file) if file != 'bank' else all_data['name'].index('bankmarketing')
    orig_data = all_data['data'][names_index]
    index = indexes[names_index]
    data_idx = list(set(list(orig_data.index)) - set(index))
    data = orig_data.iloc[data_idx, :]

    qis = key_vars[names_index]
    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)

    data = aux_singleouts(qis, data)
    singleouts = data.loc[data['single_out'] == 1]
    print(len(singleouts))
    
    conditions = pd.DataFrame({
        singleouts.columns[-2]: singleouts[singleouts.columns[-2]].tolist(),
        'single_out': singleouts.single_out.tolist(),
    })
    print(msg)
    technique = msg.split('_')[1]
    eps = int(msg.split('_')[2])
    bsz = int(msg.split('_')[3])
    ebd = int(msg.split('_')[4])
    if technique == "CTGAN":
        model = CTGAN(epochs=eps, batch_size=bsz,
                        embedding_dim=ebd)
    if technique == "TVAE":
        model = TVAE(epochs=eps, batch_size=bsz, embedding_dim=ebd)
    else:
        model = CopulaGAN(
            epochs=eps, batch_size=bsz, embedding_dim=ebd)

    model.fit(data)
    # new_data = model.sample(num_rows=len(singleouts))
    # model.save(f'output/data_variants/GANmodels{sep}{file}_{technique}_ep{eps}_bs{bsz}_ed{ebd}_model.pkl')
    try:
        new_data = model.sample_remaining_columns(conditions)
        # add non-singleouts
        new_data = pd.concat(
            [new_data, data.loc[data['single_out'] == 0]])
            # save synthetic data
        new_data.to_csv(
            f'{output_folder}{sep}{file}_{technique}_ep{eps}_bs{bsz}_ed{ebd}.csv',
            index=False)
    except:
        pass
