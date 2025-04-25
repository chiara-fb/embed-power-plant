import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from types import SimpleNamespace
from data.data_utils import *
from data.generate_subsequences import *



def make_market_data(df: pd.DataFrame, cfg: SimpleNamespace)-> pd.DataFrame:
    market_data = df.droplevel(1, axis=1)
    market_data["Hour"] = market_data.index.hour
    market_data["Day"] = market_data.index.day_of_week
    market_data["Year"] = market_data.index.year - market_data.index.year.min()
    market_data = pd.get_dummies(market_data, columns=["Hour", "Day", "Year"])
    market_data = market_data[cfg.regressors]
    return market_data


def make_units_data(df: pd.DataFrame, cfg: SimpleNamespace)-> pd.DataFrame:
    units_data = df["ActualGenerationOutput"][cfg.units]
    return units_data


def make_encodings(cfg: SimpleNamespace)-> torch.tensor:
    if "encodings" in cfg.__dict__:
        encodings, others_cat = cfg.encodings, cfg.encode_full_dims
    else:
        encodings, others_cat = [cfg.units], False

    one_hots = [map_encodings(name, encodings, others_cat) for name in cfg.units]
    return torch.stack(one_hots)


def df_to_minmax_tensor(df: pd.DataFrame, device: torch.device = None)-> torch.tensor:
    minmax_arr = MinMaxScaler().fit_transform(df)
    tensor = torch.tensor(minmax_arr, device=device)
    return tensor


def get_datasets(cfg: SimpleNamespace, device: torch.device = None)-> tuple[Dataset, Dataset, Dataset]:
    # Instantiate the dataset and dataloader
    df = pd.read_parquet(cfg.data_file)
    df = df[df.index.year.isin(cfg.years)]
    # clip generation to be within 1.2x the installed capacity (overgeneration is possible for "sprints")
    df["ActualGenerationOutput"] = df["ActualGenerationOutput"].clip(
        upper=df["InstalledGenCapacity"] * 1.2
    )
    market_data = make_market_data(df, cfg)
    units_data = make_units_data(df, cfg)
    market_data, units_data = df_to_minmax_tensor(market_data), df_to_minmax_tensor(
        units_data
    )
    unit_encodings = make_encodings(cfg)

    train_dataset, val_dataset, test_dataset = train_val_test_split(market_data,
                                                                       units_data,
                                                                       unit_encodings,
                                                                       cfg,
                                                                       device,
                                                                        )
    
    return train_dataset, val_dataset, test_dataset
