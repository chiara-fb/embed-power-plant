from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

def mape(x:pd.Series, mean:bool=None) -> float:
    if mean is None:
        mean = x.mean()
    mape = ((x - mean) / mean).abs().mean()
    return mape

def market_indicators(df:pd.DataFrame, cfg:SimpleNamespace)-> pd.DataFrame:
    """Compute market indicators for each year and unit in the dataframe. 
    Market indicators are: market value, market value factor, scaled generation,
    generation share and correlation"""

    #define the market value functions    
    market_value = lambda gen,price: price.dot(gen) / gen.sum()
    market_value_factor = lambda gen,price: market_value(gen,price) / price.mean()
    df['ResLoad'] = df['TotalLoadValue'] - df['TotalRESGeneration']

    market_list = []
    for year in cfg.years:
        df_year = df.loc[df.index.year == year]
        gen_year = df_year['ActualGenerationOutput'][cfg.units]
        mv_year = gen_year.apply(lambda x: market_value(x,df_year['Price']))
        mv_year = mv_year.rename(f"MarketValue_{year}")
        mvf_year = gen_year.apply(lambda x: market_value_factor(x,df_year['Price']))
        mvf_year = mvf_year.rename(f"MarketValueFactor_{year}")
        scaled_gen = (gen_year/df_year["InstalledGenCapacity"][cfg.units])
        scaled_gen = scaled_gen.mean().rename(f"ScaledGen_{year}")
        gen_share = gen_year.sum() / gen_year.sum().sum()
        gen_share = gen_share.rename(f"GenShare_{year}")
        market_list.extend([mv_year, mvf_year, scaled_gen, gen_share])

        for var in ["Price", "ResLoad"]:  
            year_corr = gen_year.corrwith(df_year[var])
            year_corr = year_corr.rename(f"{var}_{year}")
            market_list.append(year_corr)
        
    market_df = pd.concat(market_list, axis=1)
    return market_df


def heat_indicators(df:pd.DataFrame, when2heat:pd.DataFrame, cfg:SimpleNamespace, heat_type:str='MFH')-> pd.DataFrame:
    """Compute heat indicators for each year and unit in the dataframe."""
    
    #define the summer months as [apr-sep] and winter months as [oct-mar]
    summer_months = [4,5,6,7,8,9]
    winter_months = [10,11,12,1,2,3]

    heat_list = []
    for year in cfg.years:
        df_year = df.loc[df.index.year == year]
        gen_year = df_year['ActualGenerationOutput'][cfg.units]
        scaled_gen = (gen_year/df_year["InstalledGenCapacity"][cfg.units])
        #summer generation
        summer_gen = scaled_gen.loc[scaled_gen.index.month.isin(summer_months)]
        summer_gen = summer_gen.mean().rename(f"SummerGen_{year}") 
        #winter generation
        winter_gen = scaled_gen.loc[scaled_gen.index.month.isin(winter_months)]
        winter_gen = winter_gen.mean().rename(f"WinterGen_{year}") 
        #winter-summer avg generation difference
        winsum_diff = winter_gen - summer_gen
        winsum_diff.name = f"Winter-Summer_{year}"
        #correlation with heat demand
        when2heat_year = when2heat.loc[when2heat.index.year == year]    
        when2heat_corr = gen_year.corrwith(when2heat_year[f"DE_heat_profile_space_{heat_type}"])
        when2heat_corr = when2heat_corr.rename(f"{heat_type}_{year}")

        heat_list.extend([summer_gen, winter_gen, winsum_diff, when2heat_corr])

    heat_df = pd.concat(heat_list, axis=1)
    return heat_df



def compute_max_ramp_rate(df:pd.DataFrame, units:list) -> pd.Series:
    """
    Computes the maximum ramp rate for each asset in the dataframe.
    :param df: DataFrame containing asset data
    :return: DataFrame with maximum ramp rates
    """
    gen = df['ActualGenerationOutput'][units]
    ramp = (gen - gen.shift(1)).abs().max() / 60
    ramp = ramp.rename('Max Ramp (MW/min)')
    ramp.index.name = 'EIC Code'
    return ramp


    
def cluster_statistics(clusters:pd.Series, stats_file:str, cfg:SimpleNamespace, savefig:bool=True)-> tuple:
    """Compute the number of elements in each cluster and the number of elements in each class
    """
    stats = pd.read_excel(stats_file, index_col=0)
    # assert len(clusters) == len(stats), "Number of clusters and units in the stats_file must be equal"

    stats["Sw/industry"] = stats["Owner short"].str.contains("(sw|wchem|thys|hkm|pck|bask)", regex=True)
    all_stats = clusters.to_frame().join(stats)
    ramp_rate = compute_max_ramp_rate(all_stats, cfg.units)
    all_stats = all_stats.join(ramp_rate)

    stats_cols = (["Free market", "CHP", "Sw/industry", "Commissioning year", "Installed Capacity", "Max Ramp (MW/min)"])
    
    #compute within-cluster averages and MAPE (Mean Absolute Percentage Deviation) for each cluster
    clust_size = all_stats.groupby('clusters').size().rename("Cluster size")
    cluster_avg = all_stats.groupby('clusters')[stats_cols].mean()   
    cluster_mape = all_stats.groupby('clusters')[stats_cols].agg(mape)

    #compute sample averages and between-cluster MAPE (Mean Absolute Percentage Deviation)
    sample_avg = all_stats[stats_cols].mean().rename("Sample avg")
    sample_avg["Cluster size"] = len(stats)/len(clust_size)
    mape_avg = mape(cluster_avg, sample_avg).rename("MAPE")
    mape_avg["Cluster size"] = mape(clust_size, len(stats)/len(clust_size))

    #add cluster sizes to the dataframes
    cluster_mape = pd.concat([clust_size, cluster_mape], axis=1)
    cluster_avg = pd.concat([clust_size, cluster_avg], axis=1)

    #add between-cluster MAPE (Mean Absolute Percentage Deviation)
    cluster_avg = pd.concat([cluster_avg, 
                            sample_avg.to_frame().T,
                            mape_avg.to_frame().T], axis=0)

    
    #final dataframes of within-cluster averages and MAPE (Mean Absolute Percentage Deviation)
    return cluster_avg, cluster_mape