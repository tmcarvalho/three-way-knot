"""_summary_
"""
# %%
import gc
import sys
import pandas as pd
import numpy as np
from record_linkage import threshold_record_linkage
sys.path.append('./')
from dataprep.preprocessing import get_indexes, quasi_identifiers, read_data

def aux_singleouts(key_vars, data):
    """Calculates single outs using k-anonymity measure

    Args:
        key_vars (list[str]): set of quasi-identifiers
        dt (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with single outs concerning a set of quasi-identifiers
    """
    k = data.groupby(key_vars)[key_vars[0]].transform(len)
    data['single_out'] = None
    data['single_out'] = np.where(k == 1, 1, 0)
    data = data[data['single_out'] == 1]
    return data


def privacy_risk(transf_file, orig_data, args, key_vars):
    dict_per = {'privacy_risk_50': [], 'privacy_risk_70': [],
                'privacy_risk_90': [], 'privacy_risk_100': [], 'ds': []}

    transf_data = pd.read_csv(f'{args.input_folder}/{transf_file}')

    print(transf_file)
    len_data = len(transf_data)
    # select single outs in the original data according the set of key vars
    orig_data = aux_singleouts(key_vars, orig_data)
    transf_data = transf_data.loc[transf_data.single_out==1]
    
    percentages = threshold_record_linkage(
        transf_data,
        orig_data,
        key_vars,
        len_data)

    dict_per['privacy_risk_50'].append(percentages[0])
    dict_per['privacy_risk_70'].append(percentages[1])
    dict_per['privacy_risk_90'].append(percentages[2])
    dict_per['privacy_risk_100'].append(percentages[3])
    dict_per['ds'].append(transf_file.split('.csv')[0])
    del percentages
    gc.collect()

    return dict_per


def apply_rl(file, args):
    """Prepare the data to apply record linkage

    Args:
        file (str): Transformation file's name
        args (args)
    """
    # get original data
    data = read_data()
    indexes = get_indexes()
    names_index = data['name'].index(file.split("_")[0]) if file.split(
        "_")[0] != 'bank' else data['name'].index('bankmarketing')
    orig_data = data['data'][names_index]
    index = indexes[names_index]
    idx = list(set(list(orig_data.index)) - set(index))
    orig_data = orig_data.iloc[idx, :].reset_index(drop=True)

    # get quasi-identifiers
    all_key_vars = quasi_identifiers()
    key_vars = all_key_vars[names_index]

    print(key_vars)
    risk = privacy_risk(file, orig_data, args, key_vars)
    total_risk = pd.DataFrame.from_dict(risk)
    total_risk.to_csv(f'{args.output_folder}/{file}', index=False)
    del risk
    gc.collect()
