"""This script will modeling the data variants
"""
# pylint: disable=import-error
# pylint: disable=wrong-import-position
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from models import evaluate_model
sys.path.append('./')
from dataprep.preprocessing import read_data, get_indexes, sensitive_attributes
from fairmodels.fairlearn import evaluate_fairlearn
from fairmodels.fairmask import evaluate_fairmask
# from fairmodels.fairgbm_classifier import evaluate_fairgbm

def save_results(file, args, results):
    """Create a folder if dooes't exist and save results

    Args:
        file (string): file name
        args (args): command line arguments
        results (list of Dataframes): results for cross validation and out of sample
    """
    if args.fairtype!='fairgbm':
        output_folder_val = (
            f'{args.output_folder}/validation')
        if not os.path.exists(output_folder_val):
            os.makedirs(output_folder_val)
        results[0].to_csv(f'{output_folder_val}/{file}', index=False)
    
    output_folder_test = (
        f'{args.output_folder}/test')
    if not os.path.exists(output_folder_test):
        os.makedirs(output_folder_test)
    results[1].to_csv(f'{output_folder_test}/{file}', index=False)

# %%


def priv_groups(x_train, x_test, file):
    protected_classes = {'ds_name': ['adult','german','bankmarketing'],
                                'privileged_range_min': [25, 25, 25, 17],
                                'privileged_range_max': [60, 26, 60, 18]}

    if 'age' in x_train.columns and file in protected_classes['ds_name']:
        pc_idx = protected_classes['ds_name'].index(file)
        x_train.loc[:, 'age'] = x_train.apply(lambda x: int(protected_classes['privileged_range_min'][pc_idx] <= x['age'] <= protected_classes['privileged_range_max'][pc_idx]), axis=1)
        x_test.loc[:, 'age'] = x_test.apply(lambda x: int(protected_classes['privileged_range_min'][pc_idx] <= x['age'] <= protected_classes['privileged_range_max'][pc_idx]), axis=1)
    if file in ['adult', 'compas', 'diabets', 'lawschool']:
        value = Counter(x_train.race).most_common()[0]
        x_train.loc[:, 'race'] = np.where(x_train['race']==value[0], 1, 0)
        x_test.loc[:, 'race'] = np.where(x_test['race']==value[0], 1, 0)
    if 'bankmarketing' in file:
        value = Counter(x_train.marital).most_common()[0]
        x_train.loc[:, 'marital'] = np.where(x_train['marital']==value[0], 1, 0)
        x_test.loc[:, 'marital'] = np.where(x_test['marital']==value[0], 1, 0)
    if 'credit' in file:
        value_edu = Counter(x_train.EDUCATION).most_common()[0]
        value_mar = Counter(x_train.MARRIAGE).most_common()[0]
        x_train.loc[:, 'EDUCATION'] = np.where(x_train['EDUCATION']==value_edu[0], 1, 0)
        x_test.loc[:, 'EDUCATION'] = np.where(x_test['EDUCATION']==value_edu[0], 1, 0)
        x_train.loc[:, 'MARRIAGE'] = np.where(x_train['MARRIAGE']==value_mar[0], 1, 0)
        x_test.loc[:, 'MARRIAGE'] = np.where(x_test['MARRIAGE']==value_mar[0], 1, 0)
    if 'ricci' in file:
        value = Counter(x_train.Race).most_common()[0]
        x_train.loc[:, 'Race'] = np.where(x_train['Race']==value[0], 1, 0)
        x_test.loc[:, 'Race'] = np.where(x_test['Race']==value[0], 1, 0)

    return x_train, x_test
# %%


def modeling_data(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file
    """
    print(f'{args.input_folder}/{file}')
    if file.split("_")[0] not in ['loans', 'students']:
        indexes = get_indexes()
        all_data = read_data()
        if args.datatype != 'Original':
            names_index = all_data['name'].index(file.split(
            "_")[0]) if file.split("_")[0] != 'bank' else all_data['name'].index('bankmarketing')
        else:
            names_index = all_data['name'].index(file.split(".")[0])
        index = indexes[names_index]

        # get original data to extract test sample
        test_data = all_data['data'][names_index]
        # data to modeling
        sep = ',' if file.split(".")[0] != 'compas' else ';'
        data = pd.read_csv(f'{args.input_folder}/{file}', sep=sep) if args.datatype != 'Original' else test_data.copy()

        # get sensitive attributes
        all_sa = sensitive_attributes()
        set_sa = all_sa[names_index]

        # prepare data to modeling
        test_data = test_data.apply(LabelEncoder().fit_transform)
        if file.split("_")[0] == 'diabets': # fix issue with '<'
            data['diag_1'] = LabelEncoder().fit_transform(data['diag_1'].astype(str))
            data['diag_2'] = LabelEncoder().fit_transform(data['diag_2'].astype(str))
            data['diag_3'] = LabelEncoder().fit_transform(data['diag_3'].astype(str))
            data['readmitted'] = LabelEncoder().fit_transform(data['readmitted'].astype(str))
            data['A1Cresult'] = LabelEncoder().fit_transform(data['A1Cresult'].astype(str))
        data = data.apply(LabelEncoder().fit_transform)

        if args.datatype == 'Original':
            idx = list(set(data.index.tolist()) - set(index))
            x_train, y_train = data.iloc[idx, :-1], data.iloc[idx, -1]
        else:
            x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]

        x_test = test_data.iloc[index, :-1]
        y_test = test_data.iloc[index, -1]

        if args.fairtype == 'fairlearn':
            if file.split('.')[0].split('_')[0] not in ['dutch', 'heart']:
                x_train, x_test = priv_groups(x_train, x_test, file.split('.')[0].split('_')[0])
            results = evaluate_fairlearn(
                x_train, x_test, y_train, y_test, set_sa)
        elif args.fairtype == 'fairmask':
            if file.split('.')[0].split('_')[0] not in ['dutch', 'heart']:
                x_train, x_test = priv_groups(x_train, x_test, file.split('.')[0].split('_')[0])
            try:
                results = evaluate_fairmask(
                    x_train, x_test, y_train, y_test, set_sa)
            except:
                pass
        # elif args.fairtype == 'fairgbm':
        #     if file.split('.')[0].split('_')[0] not in ['dutch', 'heart']:
        #         x_train, x_test = priv_groups(x_train, x_test, file.split('.')[0].split('_')[0])
        #     results = evaluate_fairgbm(
        #         x_train, x_test, y_train, y_test, set_sa)
        else:
            #if file.split('.')[0].split('_')[0] not in ['dutch', 'heart']:
            _, x_test_sa = priv_groups(x_train, x_test, file.split('.')[0].split('_')[0])
            results = evaluate_model(x_train, x_test, y_train, y_test, x_test_sa[set_sa])
        try:
            save_results(file, args, results)
        except: pass
