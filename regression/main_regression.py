import pandas as pd
from sklearn.metrics import mean_squared_error
from types import SimpleNamespace
from linearmodels.panel import PanelOLS, compare
import argparse
import json

import os
import sys

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

def make_dummy(df, dummy_name):
    """Returns a dummy variable for a given time attribute of the dataframe index"""
    dummy_value = getattr(df.index, dummy_name)
    dummy = pd.get_dummies(dummy_value, prefix=dummy_name, drop_first=True, dtype=int)
    dummy = dummy.set_index(df.index)
    return dummy

def read_input_data(cfg:SimpleNamespace) -> tuple:
    #read datasets, resample to config frequency (daily, weekly, monthly)
    df = pd.read_parquet(cfg.data_file)
    df = df.resample(cfg.freq).mean()
    df.index.name ="DateTime"
    #read when2heat, resample to config frequency (daily, weekly, monthly)

    if "HeatDemand" in cfg.regressors:
        when2heat = pd.read_parquet(cfg.when2heat_file)
        when2heat = when2heat['DE_heat_profile_space_MFH'].rename('HeatDemand')
        when2heat.index = pd.to_datetime(when2heat.index, utc=False)
        when2heat = when2heat.resample(cfg.freq).mean()
    else:
        when2heat = None

    return df, when2heat

def make_target(target:pd.Series, 
                units:list,
                name:str="Y") -> pd.Series:
    
    #make target variable as panel data: stack the dataframe
    #and keep only the clusters that are in the clusters dataframe
    # join the clusters to the target variable
    
    target = target.stack().rename(name).to_frame()
    target.index = target.index.swaplevel()
    target.index.names = ['EIC Code','DateTime']
    target = target[target.index.get_level_values('EIC Code').isin(units)]
    return target

def make_regressors(df:pd.DataFrame, 
                    cfg:SimpleNamespace, 
                    when2heat:pd.Series) -> pd.DataFrame:
    
    #join heat demand if needed, keep regressors only
    #scale regressors for better interpretability
    #add time dummies
    
    if 'HeatDemand' in cfg.regressors:
        X = df.droplevel(axis=1, level=1).join(when2heat, how='left')[cfg.regressors]
    else:
        X = df[cfg.regressors].droplevel(axis=1, level=1)

    for regr, scale in cfg.scale_regressors.__dict__.items():
        X[regr] = X[regr] / scale
    
    for regr in cfg.first_diff_regressors:
        X[regr] = X[regr] - X[regr].shift(1).bfill()
    
    time_dummies = pd.concat([make_dummy(X, dummy) 
                        for dummy in cfg.time_regressors], axis=1)

    X = X.join(time_dummies, on='DateTime')  
    return X


def make_main_model(cfg:SimpleNamespace) -> tuple:
    #make function to make the regression comparison

    df, when2heat = read_input_data(cfg)
    Y_flat = df['ActualGenerationOutput'] / df['ActualGenerationOutput'].max()
    Y = make_target(Y_flat, cfg.units, name="Y")

    if cfg.lagged_regressor:
        Y_lag_flat = Y_flat.shift(1).bfill()
        Y_lag = make_target(Y_lag_flat, cfg.units, "Y_lag")
        Y = Y.join(Y_lag)

        if cfg.first_diff_target:
            Y = Y['Y'] - Y['Y_lag']
    
    X = make_regressors(df, cfg, when2heat)
    regr_df = Y.join(X, on='DateTime') 
    regr_df = regr_df.dropna(how='any')
    regr_df = regr_df.sort_index()

    train_end = None if cfg.train_end is None else pd.Timestamp(cfg.train_end)
    test_end = None if cfg.test_end is None else pd.Timestamp(cfg.test_end)
    
    train = regr_df.loc[:, :train_end, :]
    test = regr_df.loc[:, train_end:test_end, :]
 
    X_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
    X_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]
    
    
    return (X_train, y_train), (X_test, y_test)



def make_regression(cfg:SimpleNamespace, 
                    csv_path:str, 
                    tex_path:str):

    #comparison_dict, mse_dict = {}, {}
    cluster_data = make_main_model(cfg)

    
    (X_train, y_train), (X_test, y_test) = cluster_data
    model = PanelOLS(y_train, X_train, entity_effects=cfg.entity_effects)
    results = model.fit()
    pred_train, pred_test = results.predict(X_train), results.predict(X_test)
    mse_tuple = (mean_squared_error(y_train, pred_train),
                    mean_squared_error(y_test, pred_test))
    
    
    comparison = compare({'main':results}, precision='std_errors', stars=True)
    summary = comparison.summary
    mse_df = pd.Series(data=mse_tuple, index=['train', 'test'])

    with open(csv_path, "w") as csv_file:
        csv_file.write(summary.as_csv())
        csv_file.write(mse_df.to_csv())
        
    with open(tex_path, "w") as tex_file:
        tex_file.write(summary.as_latex())
        tex_file.write(mse_df.to_latex())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--csv_path", type=str, default="main_comparison.csv")
    parser.add_argument("--tex_path", type=str, default="main_comparison.txt")
    args = parser.parse_args()

    with open(args.config) as cfg_file:
        cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
    
    make_regression(cfg, args.csv_path, args.tex_path)