
import torch
import joblib
import numpy as np 
import os 
import pandas as pd
from tqdm import tqdm

import torch
import pandas as pd 
import numpy as np
import ast
import joblib
import os
import sys
from math import gcd
from scipy.signal import filtfilt, resample_poly
from fractions import Fraction
import argparse


def get_data_info(dataset_name, prefix="", usecolumns=None):
    """
    This function returns meta data about the dataset such as user/ppg dataframes,
    column name of user_id, and the raw ppg directory.

    Args:
        dataset_name (string): string for selecting the dataset
        prefix (string): prefix for correct path
        usecolumns (list): quick loading if the .csv files contains many columns or if > 0.5GB

    Returns:
        df_train (pandas.DataFrame): training dataframe containing user id and segment id 
        df_val (pandas.DataFrame): validation dataframe containing user id and segment id 
        df_test (pandas.DataFrame): test dataframe containing user id and segment id 
        case_name (string): column name containing user id
        path (string): path to ppg directory
    """

    if dataset_name == "mesa":
        case_name = "mesaid"
        path = f"{prefix}../data/mesa/mesappg/"

        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/mesa/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/mesa/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/mesa/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'mesaid'] = df_train.mesaid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'mesaid'] = df_val.mesaid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'mesaid'] = df_test.mesaid.apply(lambda x: str(x).zfill(4))
        
    if dataset_name == "vital":
        path = f"{prefix}../data/vitaldbppg/"
        case_name = "caseid"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/vital/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/vital/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/vital/test_clean.csv", usecols=usecols)

        df_train.loc[:, 'caseid'] = df_train.caseid.apply(lambda x: str(x).zfill(4))
        df_val.loc[:, 'caseid'] = df_val.caseid.apply(lambda x: str(x).zfill(4))
        df_test.loc[:, 'caseid'] = df_test.caseid.apply(lambda x: str(x).zfill(4))

    if dataset_name == "mimic":
        case_name = "SUBJECT_ID"
        path = f"{prefix}../data/mimic/ppg" # 1 stage of filtered data 
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        df_train = pd.read_csv(f"{prefix}../data/mimic/train_clean.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/mimic/val_clean.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/mimic/test_clean.csv", usecols=usecols)


    if dataset_name == "sdb":
        case_name = "subjectNumber"
        path = f"{prefix}../data/sdb/ppg"  
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/sdb/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/sdb/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/sdb/test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "ppg-bp":
        case_name = "subject_ID"
        path = f"{prefix}../data/ppg-bp/ppg"  
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 

        df_train = pd.read_csv(f"{prefix}../data/ppg-bp/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/ppg-bp/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/ppg-bp/test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "ecsmp":
        case_name = "ID"
        path = f"{prefix}../data/ecsmp/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
    
    if dataset_name == "wesad":
        case_name = "subjects"
        path = f"{prefix}../data/wesad/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)
    
    if dataset_name == "dalia":
        case_name = "subjects"
        path = f"{prefix}../data/dalia/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)
    
    if dataset_name == "marsh":
        case_name = "subjects"
        path = f"{prefix}../data/marsh/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(4))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(4))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(4))
        
    if dataset_name == "numom2b":
        case_name = "subjects"
        path = f"{prefix}../data/numom2b/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)
    
    if dataset_name == "bidmc":
        case_name = "subjects"
        path = f"{prefix}../data/bidmc/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)

        df_train.loc[:, case_name] = df_train[case_name].apply(lambda x:str(x).zfill(2))
        df_val.loc[:, case_name] = df_val[case_name].apply(lambda x:str(x).zfill(2))
        df_test.loc[:, case_name] = df_test[case_name].apply(lambda x:str(x).zfill(2))
    
    if dataset_name == "mimicAF":
        case_name = "subjects"
        path = f"{prefix}../data/mimicAF/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)
    
    if dataset_name == "vv":
        case_name = "subjects"
        path = f"{prefix}../data/vv/ppg"
        if usecolumns is not None:
            usecols = np.concatenate([[case_name], usecolumns])
        else:
            usecols = None 
        
        df_train = pd.read_csv(f"{prefix}../data/{dataset_name}/train.csv", usecols=usecols)
        df_val = pd.read_csv(f"{prefix}../data/{dataset_name}/val.csv", usecols=usecols)
        df_test = pd.read_csv(f"{prefix}../data/{dataset_name}/test.csv", usecols=usecols)

    return df_train, df_val, df_test, case_name, path


def get_data_for_ml(df, dict_embeddings, case_name, label, level="patient"):
    """
    Extract features and label starting from the dictionary
    
    Args:
        df (pandas.DataFrame): dataframe containing user id, etc.
        dict_embeddings (dictionary): dictionary containing extracted embeddings
        case_name (string): column name for user id in dataframe 
        label (string): label to extract
        level (string): patient, averages value of segments for a user
    
    Returns:
        X (np.array): feature array
        y (np.array): label array
        keys (list): test keys

    """
    y = []
    if level == "patient":
        df = df.drop_duplicates(subset=[case_name])

    for key in dict_embeddings.keys():
        if level == "patient":
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values[0])
        else:    
            y.append(df[df.loc[:, case_name] == key].loc[:, label].values)
    X = np.vstack([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings.values()])
    y = np.hstack(y)
    return X, y, list(dict_embeddings.keys())

def get_data_for_ml_from_df(df, dict_embeddings, case_name, label, level="patient"):
    """
    Extract features and label starting from the dataframe

    Args:
        df (pandas.DataFrame): dataframe containing user id, etc.
        dict_embeddings (dictionary): dictionary containing extracted embeddings
        case_name (string): column name for user id in dataframe 
        label (string): label to extract
        level (string): patient, averages value of segments for a user
    
    Returns:
        X (np.array): feature array
        y (np.array): label array
        keys (list): test keys
    """
    X = []
    y = []
    df = df.drop_duplicates(subset=[case_name])
    filenames = df[case_name].values
    for f in filenames:
        if f in dict_embeddings.keys():
            if level == "patient":
                y.append(df[df.loc[:, case_name] == f].loc[:, label].values[0])
            else:    
                y.append(df[df.loc[:, case_name] == f].loc[:, label].values)
            X.append([k.cpu().detach().numpy() if type(k) == torch.Tensor else k for k in dict_embeddings[f]])
    X = np.vstack(X)
    return X, np.array(y), filenames

def extract_labels(y, label, binarize_val = None):

    """
    The raw labels are converted to categorical for classification

    Args:
        y (np.array): label array in raw form 
        label (string) :label name
        binarize_val: Use the median to binarize the label
    
    Returns:
        y (np.array): label array ready for trianing/eval
    """
    
    if label == "age":
        y = np.where(y > 50, 1, 0)
    
    if label == "sex":
        y = np.where(y == "M", 1, 0)
    
    if label in ['bmi', 'es', 'cr', 'TMD']:
        y = np.where(y > binarize_val, 1, 0)
    
    if label == "icu_days":
        y = np.where(y > 0, 1, 0)
    
    if label == "death_inhosp":
        y = y

    if label == "optype":
        dict_label = {'Colorectal': 0,
        'Biliary/Pancreas': 1,
        'Stomach': 2,
        'Others': 3,
        'Major resection': 4,
        'Minor resection': 5,
        'Breast': 6,
        'Transplantation': 7,
        'Thyroid': 8,
        'Hepatic': 9,
        'Vascular': 10}
        y = [dict_label[op] for op in y]
    
    if label == "AHI":
        y = np.where(y > 0, 1, 0)
    
    if label == "Hypertension":
        y = np.where(y == "Normal", 0, 1)
    
    if label == "Diabetes" or label == "cerebrovascular disease" or label == "cerebral infarction":
        y = np.where(y == "0", 0, 1)
    
    if label == "valence" or label == "arousal":
        y = np.where(y <= 5, 1, 0)

    if label == "affect":
        y = y
    
    if label == "activity":
        y = y
    
    if label == "nsrr_current_smoker" or label == "nsrr_ever_smoker":
        y = np.where(y == "yes", 1, 0)
    
    if label == "sds":
        y = np.where(y > 49, 1, 0)
    
    if label == "DOD":
        y = np.where(pd.notna(y), 1, 0)
    
    if label == "stdyvis":
        y = np.where(y == 3, 1, 0)
    
    if label == "afib":
        y = np.where(y == "af", 1, 0)

    return y


def bootstrap_metric_confidence_interval(y_test, y_pred, metric_func, num_bootstrap_samples=500, confidence_level=0.95):
    bootstrapped_metrics = []

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # Bootstrap sampling
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        indices = np.random.choice(range(len(y_test)), size=len(y_test), replace=True)
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]

        # Calculate the metric for the resampled data
        metric_value = metric_func(y_test_sample, y_pred_sample)
        bootstrapped_metrics.append(metric_value)

    # Calculate the confidence interval
    lower_bound = np.percentile(bootstrapped_metrics, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_metrics, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound, bootstrapped_metrics

def sanitize(arr):
    """
    Convert an list/array from a string to a float array
    """
    parsed_list = ast.literal_eval(arr)
    return np.array(parsed_list, dtype=float)

def load_model(model, filepath):
    """
    Load a PyTorch model from a specified file path.

    Args:
    model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
    filepath (str): The path from which the model will be loaded.

    Returns:
    model (torch.nn.Module): The model with the loaded state dictionary.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

def batch_load_signals(path, case, segments):
    """
    Load ppg segments in batches
    """
    batch_signal = []
    for s in segments:
        batch_signal.append(joblib.load(os.path.join(path, case, str(s))))
    return np.vstack(batch_signal)

def load_model_without_module_prefix(model, checkpoint_path, device='cpu'):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # Create a new state_dict with the `module.` prefix removed
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # Remove `module.` prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load the new state_dict into the model
    model.load_state_dict(new_state_dict)

    return model

def resample_batch_signal(X, fs_original, fs_target, axis=-1):
    """
    Apply resampling to a 2D array with no of segments x values

    Args:
        X (np.array): 2D segments x values array
        fs_original (int/float): Source frequency 
        fs_target (int/float): Target frequency
        axis (int): index to apply the resampling.
    
    Returns:
        X (np.array): Resampled 2D segments x values array
    """
    # Convert fs_original and fs_target to Fractions
    fs_original_frac = Fraction(fs_original).limit_denominator()
    fs_target_frac = Fraction(fs_target).limit_denominator()
    
    # Find the least common multiple of the denominators
    lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
    
    # Scale fs_original and fs_target to integers
    fs_original_scaled = fs_original_frac * lcm_denominator
    fs_target_scaled = fs_target_frac * lcm_denominator
    
    # Calculate gcd of the scaled frequencies
    gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
    
    # Calculate the up and down factors
    up = fs_target_scaled.numerator // gcd_value
    down = fs_original_scaled.numerator // gcd_value
    
    # Perform the resampling
    X = resample_poly(X, up, down, axis=axis)
    
    return X

def convert_keys_to_strings(d):
    return {str(k).zfill(4): v for k, v in d.items()}


def load_linear_probe_dataset_objs(dataset_name, model_name, label, func, content, level, string_convert=True, classification=True, concat=True, prefix="../"):
    
    df_train, df_val, df_test, case_name, _ = get_data_info(dataset_name=dataset_name, prefix=prefix, usecolumns=[label])
    
    if string_convert:
        dict_train = convert_keys_to_strings(joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_train{content}.p"))
        dict_val = convert_keys_to_strings(joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_val{content}.p"))
        dict_test = convert_keys_to_strings(joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_test{content}.p"))
    else:
        dict_train = joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_train{content}.p")
        dict_val = joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_val{content}.p")
        dict_test = joblib.load(f"{prefix}../data/{dataset_name}/features/{model_name}/dict_test{content}.p")

    binarize_val = None
    if label in ['bmi', 'es', 'cr', 'TMD']:
        binarize_val = np.median(df_train.loc[:, label].values)
    
    X_train, y_train, train_keys = func(df=df_train, 
                            dict_embeddings=dict_train,
                            case_name=case_name,
                            label=label,
                            level=level)
    X_val, y_val, val_keys  = func(df=df_val, 
                                dict_embeddings=dict_val, 
                                case_name=case_name,
                                label=label,
                                level=level)
    
    X_test, y_test, test_keys = func(df=df_test, 
                                dict_embeddings=dict_test, 
                                case_name=case_name,
                                label=label,
                                level=level)
    if classification:
        y_train = extract_labels(y=y_train, 
                                label=label,
                                binarize_val=binarize_val)
        y_val = extract_labels(y=y_val, 
                            label=label,
                            binarize_val=binarize_val)
        y_test = extract_labels(y=y_test, 
                                label=label,
                                binarize_val=binarize_val)
    if concat:
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
        
        return X_train, y_train, X_test, y_test, train_keys, val_keys, test_keys
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test, train_keys, val_keys, test_keys

def load_linear_probe_dataset_combined(model_name, label, func, content, level, classification=True, concat=True, prefix="../"):
    
    if concat:
        X_train_vital, y_train_vital, X_test_vital, y_test_vital, _, _, _ = load_linear_probe_dataset_objs(dataset_name='vital', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)
        X_train_mesa, y_train_mesa, X_test_mesa, y_test_mesa, _, _, _ = load_linear_probe_dataset_objs(dataset_name='mesa', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)
        X_train_mimic, y_train_mimic, X_test_mimic, y_test_mimic, _, _, _ = load_linear_probe_dataset_objs(dataset_name='mimic', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)

        X_train = np.vstack((X_train_vital, X_train_mesa, X_train_mimic))
        y_train = np.concatenate((y_train_vital, y_train_mesa, y_train_mimic))
        X_test = np.vstack((X_test_vital, X_test_mesa, X_test_mimic))
        y_test = np.concatenate((y_test_vital, y_test_mesa, y_test_mimic))

        shuffle_idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

        return X_train, y_train, X_test, y_test
    else:
        X_train_vital, y_train_vital, X_val_vital, y_val_vital, X_test_vital, y_test_vital, _, _, _ = load_linear_probe_dataset_objs(dataset_name='vital', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)
        X_train_mesa, y_train_mesa, X_val_mesa, y_val_mesa, X_test_mesa, y_test_mesa, _, _, _ = load_linear_probe_dataset_objs(dataset_name='mesa', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)
        X_train_mimic, y_train_mimic, X_val_mimic, y_val_mimic, X_test_mimic, y_test_mimic, _, _, _ = load_linear_probe_dataset_objs(dataset_name='mimic', model_name=model_name, label=label, func=func, content=content, classification=classification, level=level, concat=concat, prefix=prefix)

        X_train = np.vstack((X_train_vital, X_train_mesa, X_train_mimic))
        y_train = np.concatenate((y_train_vital, y_train_mesa, y_train_mimic))
        X_val = np.vstack((X_val_vital, X_val_mesa, X_val_mimic))
        y_val = np.concatenate((y_val_vital, y_val_mesa, y_val_mimic))
        X_test = np.vstack((X_test_vital, X_test_mesa, X_test_mimic))
        y_test = np.concatenate((y_test_vital, y_test_mesa, y_test_mimic))

        shuffle_idx_train = np.random.permutation(len(X_train))
        X_train, y_train = X_train[shuffle_idx_train], y_train[shuffle_idx_train]

        return X_train, y_train, X_val, y_val, X_test, y_test

def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')