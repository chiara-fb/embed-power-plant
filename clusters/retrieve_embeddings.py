from sklearn.preprocessing import Normalizer
from data.data_utils import map_encodings
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace
from typing import IO


def model_file_to_embeddings(
    f: IO, cfg: SimpleNamespace, normalized: bool = False
) -> pd.DataFrame:
    """Load a model_dict and extract the unit embeddings, normalizing them.
    Returns a DataFrame with the embeddings, indexed by the unit EIC codes.
    Parameters:
        f[str]: Path to the model file.
    Returns:
        emb_param[pd.DataFrame]: DataFrame of normalized embedding,
        where each column is a unit.
    """
    with open(f, "rb") as f:
        model_dict = torch.load(f, map_location=torch.device("cpu"))

    for name in [
        "embedder",
        "encoder.emb_layer.weight",
        "decoder_cell.emb_layer.weight",
    ]:
        if name in model_dict:
            emb_param = model_dict[name]
            break

    emb_param = emb_param.detach().numpy()
    emb_param = emb_param.T if "emb_layer" in name else emb_param
    if normalized:
        emb_param = Normalizer("l2").fit_transform(emb_param)

    if "encodings" in cfg.__dict__:
        columns = [
            item
            for i, el in enumerate(cfg.encodings)
            for item in (el + [f"f{i}:other"] if cfg.encode_full_dims else el)
        ]
    else:
        columns = cfg.units

    emb_param = pd.DataFrame(emb_param.T, columns=columns)
    return emb_param


def from_features_to_cosine_dist(
    cfg: SimpleNamespace, emb_param: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the cosine similarity matrix between the embeddings of the feature-based
    encodings of the units.
    Parameters:
        cfg[SimpleNamespace]: Configuration.
        emb_param[pd.DataFrame]: DataFrame of normalized embedding,
        where each column is a feature."""

    emb = lambda n: emb_param.loc[
        :, [bool(x) for x in map_encodings(n, cfg.encodings, cfg.encode_full_dims)]
    ]
    dist = lambda x, y: (x.T.dot(y)).diagonal().mean() 
    mat = [[dist(emb(m).values, emb(n).values) for m in cfg.units] for n in cfg.units]
    cos_similarity = pd.DataFrame(mat, columns=cfg.units, index=cfg.units)
    dist_mat = 1 - cos_similarity
    return dist_mat
