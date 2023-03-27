"""Predictive performance
This script will test the predictive performance of the data sets.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference)

# %% evaluate a model


def evaluate_model(x_train, x_test, y_train, y_test, sa):
    """Evaluatation

    Args:
        x_train (pd.DataFrame): dataframe for train
        x_test (pd.DataFrame): dataframe for test
        y_train (np.int64): target variable for train
        y_test (np.int64): target variable for test
        sa (pd.DataFrame): set of sensitive attributes for prediction
    Returns:
        tuple: dictionary with validation, train and test results
    """

    validation_all = pd.DataFrame()
    seed = np.random.seed(1234)

    # initiate models
    rf = RandomForestClassifier(random_state=seed)
    booster = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=seed)
    reg = LogisticRegression(random_state=seed)

    pipe_rf = make_pipeline(rf)
    pipe_booster = make_pipeline(booster)
    pipe_reg = make_pipeline(reg)
    pipeline = [pipe_rf, pipe_booster, pipe_reg]

    # set parameterisation
    param_grid_rf = {
        'randomforestclassifier__n_estimators': [100, 250, 500],
        'randomforestclassifier__max_depth': [4, 7, 10]
    }
    param_grid_booster = {
        'xgbclassifier__n_estimators': [100, 250, 500],
        'xgbclassifier__max_depth': [4, 7, 10],
        'xgbclassifier__learning_rate': [0.1, 0.01]
    }
    param_grid_reg = {
        'logisticregression__C': np.logspace(-4, 4, 3),
        'logisticregression__max_iter': [1000000, 100000000]
    }
    param_grids = [param_grid_rf, param_grid_booster, param_grid_reg]
    # define metric functions
    scoring = {
        'gmean': make_scorer(geometric_mean_score),
        'acc': 'accuracy',
        'bal_acc': 'balanced_accuracy',
        'f1': 'f1',
        'f1_weighted': 'f1_weighted',
        'roc_auc_curve': make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True)
    }
    model = ['Random Forest', 'XGBoost', 'Logistic Regression']
    # Fit grid search to each pipeline
    print("Start modeling with CV")
    for idx, pipe in enumerate(pipeline):
        grid_search = GridSearchCV(
            pipe,
            param_grids[idx],
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2),
            scoring=scoring,
            return_train_score=True,
            refit='roc_auc_curve',
            n_jobs=-1).fit(x_train, y_train)
        # Store results from grid search
        validation = pd.DataFrame(grid_search.cv_results_)
        validation['model'] = model[idx]
        
        validation_all = validation if validation_all.empty else pd.concat(
                [validation_all, validation])
    best_model = validation_all.iloc[[validation_all['mean_test_acc'].idxmax()]].reset_index()

    # set best classifier parameters
    if best_model['model'][0] == 'Random Forest':
        best_clf = RandomForestClassifier(
            n_estimators=best_model['param_randomforestclassifier__n_estimators'][0],
            max_depth=best_model['param_randomforestclassifier__max_depth'][0],
            random_state=seed)
    elif best_model['model'][0] == 'XGBoost':
        best_clf = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=best_model['param_xgbclassifier__n_estimators'][0],
            max_depth=best_model['param_xgbclassifier__max_depth'][0],
            learning_rate=best_model['param_xgbclassifier__learning_rate'][0],
            random_state=seed)
    else:
        best_clf = LogisticRegression(
            C=best_model['param_logisticregression__C'][0],
            max_iter=best_model['param_logisticregression__max_iter'][0],
            random_state=seed)
    print(best_clf)
    best_clf.fit(x_train, y_train)

    # store predict scores
    score_cv = {
        'params': [], 'model': [],
        'test_accuracy': [], 'test_f1_weighted': [], 'test_gmean': [], 'test_roc_auc': [],
        'demographic_parity': [], 'equalized_odds': []
    }

    clf = best_clf.predict(x_test)
    score_cv['params'].append(best_model['params'][0])
    score_cv['model'].append(best_model['model'][0])
    score_cv['test_accuracy'].append(accuracy_score(y_test, clf))
    score_cv['test_f1_weighted'].append(
        f1_score(y_test, clf, average='weighted'))
    score_cv['test_gmean'].append(geometric_mean_score(y_test, clf))
    score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))
    score_cv['demographic_parity'].append(demographic_parity_difference(
        y_test, clf, sensitive_features=sa))
    score_cv['equalized_odds'].append(equalized_odds_difference(
        y_test, clf, sensitive_features=sa))

    score_cv = pd.DataFrame(score_cv)
    score_cv = score_cv.transpose()

    return [validation, score_cv]
