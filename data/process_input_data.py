import pandas as pd
from pathlib import Path
from multiprocessing.pool import ThreadPool
import json
from types import SimpleNamespace


def is_leap_year(year:int) -> bool:
    """Check if year is a leap year (bisextile)."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def process_csv(csv:str, 
                country:str="DE")->pd.DataFrame:   
    """Read csv file and return DataFrame."""
    df = pd.read_csv(csv, sep="\t")
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")

    df = df[df["MapCode"].str.startswith(country)] if "MapCode" in df.columns else df

    if "AreaTypeCode" in df.columns:
        if "CTY" in df["AreaTypeCode"].unique():
            df = df[df["AreaTypeCode"] == "CTY"]
        elif "CTA" in df["AreaTypeCode"].unique():
            df = df[df["AreaTypeCode"] == "CTA"]
        else:
            df = df[df["AreaTypeCode"] == "BZN"]
    return df


def read_entsoe_data(cfg:SimpleNamespace, 
                     folder_name:str, 
                     file_suffix:str, 
                     country:str="DE") -> pd.DataFrame:
    pathlist = []
    for year in cfg.years:
        for month in range(1, 13):
            file = "_".join([str(year), "{:02d}".format(month), file_suffix])
            file_name = Path(cfg.input_folder) / "entsoe" / folder_name / file
            pathlist.append(file_name)

    with ThreadPool(10) as pool:
        # result = pool.map(lambda x: pd.read_csv(x, sep='\t'),
        #                   pathlist, chunksize=10)
        result = pool.map(
            lambda x: process_csv(x, country=country), pathlist, chunksize=10
        )

    # df = pd.concat(result, ignore_index=True)
    df = pd.concat(result)
    return df


def read_physical_flows(cfg:SimpleNamespace, 
                        country:str="DE") -> pd.DataFrame:
    
    df = read_entsoe_data(cfg, "physical_flows", "PhysicalFlows_12.1.G.csv")
    df = df[df["InAreaTypeCode"] == "CTY"]
    df = df[(df["InMapCode"] == country) | (df["OutMapCode"] == country)]
    df.loc[df["InMapCode"] == country, "FlowValue"] *= -1
    df = df.groupby(["DateTime"])["FlowValue"].sum()
    df = df.resample("h").mean()
    df = df.rename("NetImport")
    return df


def read_unit_unavailability(cfg:SimpleNamespace, 
                             country:str="DE") -> pd.DataFrame:
    df = read_entsoe_data(
        cfg,
        "unavailability_per_plant",
        "UnavailabilityOfGenerationUnits_15.1.A_B.csv",
        country,
    )
    # Assume data to be in local time (based on conversations with ENTSO-E support)
    tz = df["TimeZone"].unique()[0]
    for col in ["StartOutage", "EndOutage", "StartTS", "EndTS", "UpdateTime"]:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def read_unit_generation(cfg:SimpleNamespace, 
                         country:str="DE") -> pd.DataFrame:
    df = read_entsoe_data(
        cfg,
        "generation_per_plant",
        "ActualGenerationOutputPerGenerationUnit_16.1.A.csv",
        country,
    )
    df = df.pivot(
        columns="GenerationUnitEIC",
        values=["ActualGenerationOutput", "InstalledGenCapacity"],
    )
    df = df.resample("h").mean()
    return df


def read_total_res_generation(cfg:SimpleNamespace, 
                              disaggregate:bool=True, 
                              country:str="DE") -> pd.DataFrame:
    df = read_entsoe_data(
        cfg,
        "aggregated_generation",
        "AggregatedGenerationPerType_16.1.B_C.csv",
        country,
    )
    df = df[df["ProductionType"].str.contains("Wind|Solar")]

    if disaggregate: #if disaggregated, return wind and solar separately
        df = df.pivot(columns="ProductionType", values="ActualGenerationOutput")
    else:
        df = df['ActualGenerationOutput'].rename("TotalRESGeneration")

    df = df.groupby("DateTime").sum()
    df = df.resample("h").mean()   
    return df


def read_load(cfg:SimpleNamespace, 
              country:str="DE") -> pd.DataFrame:
    """read load from ENTSO-E SFTP data."""
    
    df = read_entsoe_data(cfg, "load", "ActualTotalLoad_6.1.A.csv", country)
    df = df.resample("h")["TotalLoadValue"].mean()
    return df


def read_load_forecast(cfg:SimpleNamespace, 
                       country:str="DE") -> pd.DataFrame:
    
    df = read_entsoe_data(
        cfg, "load_forecast", "DayAheadTotalLoadForecast_6.1.B.csv", country
    )
    df = df.resample("h")["TotalLoadValue"].mean()
    return df.rename("TotalLoadForecast")


def read_day_ahead_prices(cfg:SimpleNamespace, 
                          country:str="DE") -> pd.Series:
    
    df = read_entsoe_data(cfg, "day_ahead_prices", "DayAheadPrices_12.1.D.csv", country)
    df = df[df["ResolutionCode"] == "PT60M"]
    return df["Price"]


def read_fuel_prices(cfg:SimpleNamespace) -> pd.DataFrame:

    fuel_prices = pd.read_pickle(Path(cfg.input_folder) / "fuel_prices" / cfg.fuel_file)
    fuel_prices = fuel_prices.rename(
        columns={
            "Coal Price [US$ per million BTU]": "Coal",
            "Gas Price [EUR/MWh]": "Gas",
            "ETS price [EUR/t]": "ETS",
        }
    )
    fuel_prices.index = fuel_prices.index.tz_localize(None)
    fuel_prices = fuel_prices[fuel_prices.index.year.isin(cfg.years)]
    fuel_prices = fuel_prices.ffill().bfill()
    return fuel_prices[["Coal", "Gas", "ETS"]]


def make_dataset(cfg:SimpleNamespace, verbose=True) -> pd.DataFrame:

    start, end = cfg.years[0], cfg.years[-1]
    index = pd.date_range(
        start=f"01-01-{start}", end=f"01-01-{end+1}", inclusive="left", freq="h"
    )

    net_import = read_physical_flows(cfg)
    print("Net import processed") if verbose else None
    da_prices = read_day_ahead_prices(cfg)
    print("Day-ahead prices processed") if verbose else None    
    fuel_prices = read_fuel_prices(cfg)
    print("Fuel prices processed") if verbose else None
    load = read_load(cfg)
    print("Load processed") if verbose else None
    load_fcst = read_load_forecast(cfg)
    print("Load forecast processed") if verbose else None
    res_gen = read_total_res_generation(cfg)
    print("RES generation processed") if verbose else None

    df = pd.concat(
        [net_import, da_prices, fuel_prices, load, load_fcst, res_gen], axis=1
    )

    df = df.reindex(index).ffill().bfill()
    df.columns = pd.MultiIndex.from_product([df.columns, [""]])
    unit_gen = read_unit_generation(cfg)
    unit_gen = unit_gen.reindex(index).fillna(0)
    df = pd.concat([df, unit_gen], axis=1)

    return df


if __name__ == "__main__":
    with open("config.json") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    df = make_dataset(cfg)
    path = f"data/{cfg.years[0]}-{cfg.years[-1]}_DE.parquet"
    df.to_parquet(path)
