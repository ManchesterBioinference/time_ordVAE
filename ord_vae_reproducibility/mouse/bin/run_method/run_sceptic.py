import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import sys
import os
import argparse
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
import xgboost as xgb
import sklearn

from sceptic import run_sceptic_and_evaluate
import scanpy as sc
import anndata as ad
import scipy


def run_on_splits(anndata, time_col, split_col, *args, n_cpus=1, hv=False):
    results_dir = './'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    unique_splits = anndata.obs[split_col].unique()

    all_fold_dfs = []
    for split in unique_splits:
        print(f"Processing fold: {split}")
        fold_df = process_fold(split, split_col, anndata, time_col, *args, n_cpus=n_cpus,
                               results_dir=results_dir)
        all_fold_dfs.append(fold_df)

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)


def process_fold(split, split_col, anndata, time_col, mode, ifold, n_cpus=1, hv=False, results_dir='./',
                 parameters=None, use_gpu=False, save_model=None, return_train=False):
    test = anndata[anndata.obs[split_col] == split].copy()
    train = anndata[anndata.obs[split_col] != split].copy()

    if hv:
        hvgs = sc.pp.highly_variable_genes(train, n_top_genes=2000, subset=True, inplace=False).index
        train = train[:, hvgs].copy()
        test = test[:, hvgs].copy()

    subFold_dir = f'{results_dir}/{split}/'
    if not os.path.exists(subFold_dir):
        os.makedirs(subFold_dir)

    if not parameters:
        if mode == "svm":
            parameters = {
                "C": [1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale"]
            }
        elif mode == "xgboost":
            parameters = {  
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.3],
                "n_estimators": [100],
                "subsample": [0.8]
            }
        else:
            raise ValueError(f"Unsupported method '{mode}' and no parameters provided.")

    # Use sparse matrices directly - no need to convert to dense
    X_train = train.X
    label_train = train.obs[time_col].values

    X_test = test.X
    label_test = test.obs[time_col].values

    # Convert labels to categorical values (shared encoding in train and test)
    lab = preprocessing.LabelEncoder()
    lab = lab.fit(label_train)

    y_train = lab.transform(label_train)
    label_list = lab.classes_

    if mode == "xgboost":
        xgb_model = xgb.XGBClassifier(
            tree_method='gpu_hist' if use_gpu else 'auto',
            gpu_id=0 if use_gpu else -1,
            objective='multi:softprob',
            num_class=len(label_list),
            eval_metric='mlogloss'
        )
        clf = GridSearchCV(xgb_model, parameters, cv=ifold)
    elif mode == "svm":
        svc = svm.SVC(probability=True)
        clf = GridSearchCV(svc, parameters, cv=ifold)
    else:
        raise ValueError(f"Unsupported method '{mode}'. Choose 'svm' or 'xgboost'.")

    clf.fit(X_train, y_train)

    # Predict directly on sparse matrices
    test.obs['sceptic_predicted'] = clf.predict(X_test)
    predict_proba = clf.predict_proba(X_test)
    test.obs['sceptic_pseudotime'] = np.sum(predict_proba * label_list, axis=1)

    if save_model is not None:
        save_model += '.pkl'
        with open(save_model, 'wb') as model_file:
            pickle.dump(clf, model_file)
        print(f"Model saved to {save_model}")
    
    test_result_df = pd.DataFrame(index = test.obs.index, data={
        'split': split,
        'psupertime': test.obs['sceptic_pseudotime'],
        'gt_window': test.obs[time_col]
    })
    test_result_df.columns = ['split', 'pred_time', 'gt_window']

    if return_train:
        # Predict on the full training set using sparse matrices directly
        X_train_full = train.X
        train.obs['sceptic_predicted'] = clf.predict(X_train_full)
        predict_proba_train = clf.predict_proba(X_train_full)
        train.obs['sceptic_pseudotime'] = np.sum(predict_proba_train * label_list, axis=1)
        train_result_df = pd.DataFrame(index = train.obs.index, data={
            'split': 'train',
            'psupertime': train.obs['sceptic_pseudotime'],
            'gt_window': train.obs[time_col]
        })
        train_result_df.columns = ['split', 'pred_time', 'gt_window']
        return_df = pd.concat([test_result_df, train_result_df], axis=0)
    else:
        return_df = test_result_df

    return return_df
    

def run_on_test_train(anndata, time_col, split_col, *args, n_cpus=1):
    results_dir = './'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    test_train_df = process_fold('test', split_col, anndata, time_col, *args, n_cpus=n_cpus,
                                  results_dir=results_dir, return_train=True)

    test_train_df.to_csv(f'{results_dir}results.csv', index=True)


def run_on_all(anndata, time_col, mode, ifold, n_cpus=1, hv=False, results_dir='./',
                 parameters=None, use_gpu=False, save_model=None):
    results_dir = './'
    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    if mode == "svm":
        parameters = {
            "C": [1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale"]
        }
    elif mode == "xgboost":
        parameters = {  
            "max_depth": [3, 5],
            "learning_rate": [0.1, 0.3],
            "n_estimators": [100],
            "subsample": [0.8]
        }

    # Use sparse matrices directly - no subsampling needed
    X_train = anndata.X
    label_train = anndata.obs[time_col].values

    # Convert labels to categorical values
    lab = preprocessing.LabelEncoder()
    lab = lab.fit(label_train)

    y_train = lab.transform(label_train)
    label_list = lab.classes_

    if mode == "xgboost":
        xgb_model = xgb.XGBClassifier(
            tree_method='gpu_hist' if use_gpu else 'auto',
            gpu_id=0 if use_gpu else -1,
            objective='multi:softprob',
            num_class=len(label_list),
            eval_metric='mlogloss'
        )
        clf = GridSearchCV(xgb_model, parameters, cv=ifold)
    elif mode == "svm":
        svc = svm.SVC(probability=True)
        clf = GridSearchCV(svc, parameters, cv=ifold)
    else:
        raise ValueError(f"Unsupported method '{mode}'. Choose 'svm' or 'xgboost'.")
    
    clf.fit(X_train, y_train)

    # Predict directly on sparse matrices
    anndata.obs['sceptic_predicted'] = clf.predict(X_train)
    predict_proba = clf.predict_proba(X_train)
    anndata.obs['sceptic_pseudotime'] = np.sum(predict_proba * label_list, axis=1)
    
    anndata.obs['split'] = 'all_data'
    kFold_test_result_df = anndata.obs.loc[:, ['split', 'sceptic_pseudotime', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="file loc of h5ad file")
    parser.add_argument("--time_col", help="column name for time info", default='exp2_centers')
    parser.add_argument("--split_col", help="column name in anndata.obs for splitting. Only required for in_dist",
                        required=False)
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits. Default is leave one out.", default="leave_one_out")
    parser.add_argument("--mode", help="sceptic mode", default="svm", type=str)
    parser.add_argument("--use_gpu", help="use gpu for training", action="store_true")
    parser.add_argument("--n_cpus", help="n cpus", default=1, type=int)
    args = parser.parse_args()

    time_col = args.time_col
    n_cpus = args.n_cpus
    ifold = 4
    mode = args.mode

    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, mode, ifold, n_cpus=args.n_cpus)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, mode, ifold, n_cpus=args.n_cpus)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, mode, ifold, n_cpus=args.n_cpus)
    elif args.experiment in ['test_train']:
        run_on_test_train(anndata, args.time_col, args.split_col, mode, ifold, n_cpus=args.n_cpus)
    else:
        raise ValueError(f"Unsupported experiment type '{args.experiment}'.")


if __name__ == "__main__":
    main()