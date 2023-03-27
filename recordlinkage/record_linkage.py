"""Record linkage
This script applies comparisons between the select QIs for all single outs.
"""
# %%
import gc
import pandas as pd
import recordlinkage


# %%


def record_linkage(transformed: pd.DataFrame,
                   original: pd.DataFrame, columns: list) -> pd.DataFrame:
    """_summary_

    Args:
        transformed (pd.DataFrame): transformed dataframe with singleouts
        original (pd.DataFrame): original dataframe with single outs
        columns (list): list of quasi-identifiers

    Returns:
        pd.Dataframe: comparison results with respective score
    """
    indexer = recordlinkage.Index()
    # indexer.full()
    block_var = original[columns].select_dtypes(
        include=['object']).columns.tolist()
    if len(block_var) > 0:
        block_var = block_var[0] if len(block_var) == 1 else block_var
        print(block_var)
        indexer.block(left_on=block_var, right_on=block_var)
        gc.collect()
    # dutch case
    elif set(['sex', 'edu_level', 'country_birth']).issubset(original.columns.tolist()):
        indexer.block(left_on=['sex', 'edu_level', 'country_birth'], right_on=[
                      'sex', 'edu_level', 'country_birth'])
        gc.collect()
    # credit case
    elif set(['SEX', 'EDUCATION']).issubset(original.columns.tolist()):
        indexer.block(left_on=['SEX', 'EDUCATION'], right_on=[
                      'SEX', 'EDUCATION'])
        gc.collect()

    else:
        indexer.full()
    candidates = indexer.index(transformed, original)
    print(len(candidates))
    compare = recordlinkage.Compare(n_jobs=-1)
    for idx, col in enumerate(columns):
        if col in transformed.columns:
            if transformed[col].dtype == 'object':
                original[col] = original[col].astype(str)
                compare.string(
                    col, columns[idx], label=columns[idx], method='levenshtein', threshold=0.7)
            else:
                compare.numeric(col, columns[idx],
                                label=columns[idx], method='gauss')

    comparisons = compare.compute(candidates, transformed, original)
    potential_matches = comparisons[comparisons.sum(axis=1) > 1].reset_index()
    potential_matches['Score'] = potential_matches.iloc[:, 2:].sum(axis=1)
    potential_matches = potential_matches[potential_matches['Score'] >=
                                          0.5*potential_matches['Score'].max()]
    del comparisons
    gc.collect()
    return potential_matches


def threshold_record_linkage(transformed_data, original_data, keys, len_data):
    """Apply record linkage and calculate the percentage of re-identification

    Args:
        transformed_data (pd.Dataframe): transformed data
        original_data (pd.Dataframe): original dataframe
        keys (list): list of quasi-identifiers
        len_data (int): number of records

    Returns:
        tuple(pd.Dataframe, list): dataframe with all potential matches,
        list with percentages of re-identification for 50%, 75% and 100% matches
    """
    potential_matches = record_linkage(transformed_data, original_data, keys)

    # get acceptable score (QIs match at least 50%)
    acceptable_score_50 = potential_matches[(potential_matches['Score'] >=
                                             0.5*potential_matches['Score'].max()) & (potential_matches['Score'] <
                                                                                      0.7*potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_50.groupby(['level_1'])[
        'level_0'].size()
    per_50 = ((1/level_1_acceptable_score.min()) * 100) / len_data

    # get acceptable score (QIs match at least 70%)
    acceptable_score_70 = potential_matches[(potential_matches['Score'] >=
                                             0.7*potential_matches['Score'].max()) & (potential_matches['Score'] <
                                                                                      0.9*potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_70.groupby(['level_1'])[
        'level_0'].size()
    per_70 = ((1/level_1_acceptable_score.min()) * 100) / len_data

    # get acceptable score (QIs match at least 90%)
    acceptable_score_90 = potential_matches[(potential_matches['Score'] >=
                                             0.9*potential_matches['Score'].max()) & (potential_matches['Score'] <
                                                                                      potential_matches['Score'].max())]
    level_1_acceptable_score = acceptable_score_90.groupby(['level_1'])[
        'level_0'].size()
    per_90 = ((1/level_1_acceptable_score.min()) * 100) / len_data

    # get max score (all QIs match)
    max_score = potential_matches[potential_matches['Score'] == len(keys)]
    # find original single outs with an unique match in oversampled data - 100% match
    level_1_max_score = max_score.groupby(['level_1'])['level_0'].size()
    per_100 = (
        len(level_1_max_score[level_1_max_score == 1]) * 100) / len_data

    del potential_matches
    gc.collect()

    return [per_50, per_70, per_90, per_100]

# %%
