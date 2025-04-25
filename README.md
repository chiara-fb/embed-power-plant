# Embed power plants


This is the companion repository to the paper "Flexibility of German gas-fired generation: evidence from clustering empirical operation".

This repository contains source data and scripts to train the encoder-model (`main.py`), and create clusters of units with related statistics, figures and regressions.

**Sources**

- `2019-2023_DE_v6.parquet`: input to the model
- `model.json`: final configuration of the encoder-decoder model
- `model.pth`: weights of the trained encoder-decoder model
- `stats.xlsx`: statistics on the generation units

**Scripts**

- `clusters/`
    - `cluster_statistics.py`: reports cluster statistics for a given set of clusters.
    - `compare_clusters.py`: evaluate clustering quality and compares pairwise clusters from different runs
    - `cosine_kmeans.py`: implementation of K-Means with cosine distance from [SKMeans](https://github.com/amberm291/SKMeans)
    - `main.py`: all-in-one command to run cluster comparisons
    - `retrieve_embeddings.py`: loads a `.pth` file with the model state and extract the unit embeddings, normalizing them.

- `data/`
    - `data_utils.py`: utils to process the data
    - `generate_subsequences.py`: splits the data into subsequences and then returns `torch.utils.Dataset` training, validation and testing datasets given a configuration 
    - `make_dataset.py`: implements the data pipeline
    - `process_input_data.py`: processes raw dataframe into a main datasource
    - `test_group_differences.py`: runs ANOVA and Chi-Square tests on differences in characteristics between clusters.

- `model/`
    - `EncoderDecoder.py`: encoder-decoder with one-hot encodings
    - `EncoderEmbedDecoder.py`: encoder-decoder with embeddings

- `regression/`
    - `clusters.json`: configuration for cluster-level regressions
    - `cluster_regression.py`: runs cluster-level regressions
    - `main_regression.py`: runs sample-level regression

- `trainer/`
    - `train_utils.py`: utils to train the model
    - `Trainer.py`: the main class object to train given a model and a training configuration

- `visualize/`
    - `visualize_clusters.py`: visualizes cluster-level statistics
    - `visualize_data.py`: visualizes the processed dataframe
    - `visualize_embeddings.py`: visualizes embeddings and their clusters

- `main.py`: trains the model according to a config file


## Notes on utilization

- The code was implemented using Python 3.12.4. The setup for the virtual environment is provided in `requirements.txt`. 
- The original model was trained on a GPU (kaggle GPU P100), runtimes may differ depending on the used device.
- For files that require argument parsing, a configuration file must be provided. The default configuration is `model.json`.




