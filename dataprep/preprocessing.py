"""This file contains the necessary pre-processing functions for the data sets.
"""
# %%
import os
import random
import pandas as pd
from sklearn.impute import SimpleImputer

# %%


def read_data() -> dict(name=str, data=pd.DataFrame):
    """Read all the data

    Returns:
        dict: dict with name of datasets and datasets values
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    curr_dir = f'/{os.path.join(*(script_dir.split("/")[:-1]))}'
    os.chdir(curr_dir)
    data_sets = {"name": [], "data": []}
    adult = pd.read_csv(f'{"input/adult.csv"}', sep=",")
    data_sets['data'].append(adult)
    german = pd.read_csv(f'{"input/german.csv"}', sep=",")
    data_sets['data'].append(german)
    dutch = pd.read_csv(f'{"input/dutch.csv"}', sep=",")
    data_sets['data'].append(dutch)
    bank_marketing = pd.read_csv(f'{"input/bankmarketing.csv"}', sep=",")
    del bank_marketing['id']
    data_sets['data'].append(bank_marketing)
    credit = pd.read_csv(f'{"input/credit.csv"}', sep=",")
    del credit['ID']
    data_sets['data'].append(credit)
    loans = pd.read_csv(f'{"input/loans.csv"}', sep=",")
    del loans['ID']
    del loans['dtir1']
    del loans['year']
    imputer = SimpleImputer()
    loans[['rate_of_interest', 'term', 'property_value', 'income', 'Interest_rate_spread', 'Upfront_charges', 'LTV']] = imputer.fit_transform(
        loans[['rate_of_interest', 'term', 'property_value', 'income', 'Interest_rate_spread', 'Upfront_charges', 'LTV']])
    imputer_cat = SimpleImputer(strategy='most_frequent')
    loans[['age', 'loan_limit', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization', 'submission_of_application']] = imputer_cat.fit_transform(
        loans[['age', 'loan_limit', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization', 'submission_of_application']])
    data_sets['data'].append(loans)
    compas = pd.read_csv(f'{"input/compas.csv"}', sep=";")
    del compas["id"]
    del compas["name"]
    del compas["first"]
    del compas["last"]
    del compas["violent_recid"]
    compas.drop(compas.columns[compas.isnull().sum()
                > 0], inplace=True, axis=1)
    data_sets['data'].append(compas)
    diabetes = pd.read_csv(f'{"input/diabets.csv"}', sep=",")
    del diabetes['encounter_id']
    del diabetes['patient_nbr']
    data_sets['data'].append(diabetes)
    heart = pd.read_csv(f'{"input/heart.csv"}', sep=",")
    data_sets['data'].append(heart)
    ricci = pd.read_csv(f'{"input/ricci.csv"}', sep=",")
    data_sets['data'].append(ricci)
    students = pd.read_csv(f'{"input/students.csv"}', sep=",")
    data_sets['data'].append(students)
    law_school = pd.read_csv(f'{"input/lawschool.csv"}', sep=",")
    data_sets['data'].append(law_school)

    names = ["adult", "german", "dutch", "bankmarketing", "credit", "loans",
             "compas", "diabets", "heart", "ricci", "students", "lawschool"]

    for name in names:
        data_sets['name'].append(name)

    return data_sets
# %%


def quasi_identifiers() -> list[list[str]]:
    """Provides the list of quasi-indentifiers for each dataset

    Returns:
        list[list[str]]: list of quasi-identifiers of each dataset
    """
    adult_qi = ["age", "gender", "race", "occupation", "native-country"]
    german_qi = ["age", "purpose", "employment-since",
                 "residence-since", "job", "sex", "foreign-worker"]
    ducth_qi = ["age", "sex", "edu_level", "country_birth"]
    bank_marketing_qi = ["age", "job", "marital", "education"]
    credit_qi = ["AGE", "SEX", "EDUCATION", "MARRIAGE"]
    loans_qi = ["age", "Gender", "income", "Region"]
    compas_qi = ["sex", "age", "race", "is_recid"]
    diabetes_qi = ["age", "gender", "race", 'time_in_hospital']
    heart_qi = ["age", "thalach", "sex", "cp"]
    ricci_qi = ["Combine", "Position", "Race"]
    students_qi = ["age", "school", "sex",
                   "address", "famsize", "Mjob", "Fjob"]
    law_school_qi = ["lsat", "male", "race"]

    qi_list = [adult_qi, german_qi, ducth_qi, bank_marketing_qi, credit_qi, loans_qi,
               compas_qi, diabetes_qi, heart_qi, ricci_qi, students_qi, law_school_qi]

    return qi_list


# %%


def sensitive_attributes() -> list[list[str]]:
    """Provides the list of sensitive attributes (SA) for each dataset

    Returns:
        list[list[str]]: list of sensitive attributes of each dataset
    """
    adult_sa = ["age", "gender", "race"]
    german_sa = ["age", "sex"]
    ducth_sa = ["sex"]
    bank_marketing_sa = ["age", "marital"]
    credit_sa = ["SEX", "EDUCATION", "MARRIAGE"]
    loans_sa = ["Gender"]
    compas_sa = ["sex", "race"]
    diabetes_sa = ["gender", "race"]
    heart_sa = ["sex"]
    ricci_sa = ["Race"]
    students_sa = ["sex", "age"]
    law_school_sa = ["male", "race"]

    sa_list = [adult_sa, german_sa, ducth_sa, bank_marketing_sa, credit_sa, loans_sa,
               compas_sa, diabetes_sa, heart_sa, ricci_sa, students_sa, law_school_sa]

    return sa_list


# %%


def get_indexes() -> list[list[int]]:
    """Create a list of indexes for the test data

    Returns:
        list[list[int]]: list of indexes
    """
    random.seed(42)
    indexes = []
    data = read_data()
    for each_data in data["data"]:
        random_idx = random.sample(
            list(each_data.index), k=int(0.2*len(each_data)))
        indexes.append(random_idx)
    return indexes
