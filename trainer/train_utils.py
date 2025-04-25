import torch
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from types import SimpleNamespace

def accuracy(output: torch.tensor, target: torch.tensor) -> torch.tensor:
    """Compute the accuracy of the model, given the output and target tensors.
    Parameters:
        output: tensor of shape [batch_size, out_seq_len, num_classes]
        target: tensor of shape [batch_size, out_seq_len, num_classes]

    Returns:
        acc: float indicating the accuracy of the model
    """
    output, target = output.flatten(end_dim=-2), target.flatten(end_dim=-2)
    output, target = output.argmax(dim=-1), target.argmax(dim=-1)
    acc = torch.sum(output == target) / target.numel()
    return acc


def balanced_accuracy(output: torch.tensor, target: torch.tensor) -> torch.tensor:
    """Compute the balanced_accuracy accuracy of the model to account for
    imbalanced datasets, given the output and target tensors.
    Parameters:
        output: tensor of shape [batch_size, out_seq_len, num_classes]
        target: tensor of shape [batch_size, out_seq_len, num_classes]
    Returns:
        bal_acc: float indicating the accuracy of the model
    """

    num_classes = output.shape[-1]
    output, target = output.flatten(end_dim=-2), target.flatten(end_dim=-2)
    output, target = output.argmax(dim=-1), target.argmax(dim=-1)
    bal_acc = []
    for cl in range(num_classes):
        num = ((output == cl) & (target == cl)).sum()
        den = (target == cl).sum()
        if den > 0:
            bal_acc.append(num / den)

    bal_acc = torch.tensor(bal_acc) / len(bal_acc)
    return bal_acc.sum()


def prettify_config(d: dict) -> str:
    """Transforms the config dictionary into a markdown table compatible
    with text logs in TensorBoard.
    Parameters:
        d: dictionary containing the configuration parameters
    Returns:
        table: string containing the markdown table"""

    header_row = "|Parameter|Value|"
    sep_row = "|---|---|"
    data_rows = [f"| {key} | {str(val)} |" for key, val in d.items()]
    table = "\n".join([header_row, sep_row] + data_rows)
    return table


def prettify_test(d: dict, num_classes: int) -> str:
    """Transforms the test dictionary containing hits ('O') and fails ('X') of
    predictions per unit and class into a markdown table compatible with text
    logs in TensorBoard.
    Parameters:
        d: dictionary containing the test predictions
    Returns:
        table: string containing the markdown table"""

    header_row = "|Unit|" + "|".join([f"Hits_{c}" for c in range(num_classes)]) + "|"
    sep_row = "|" + "|".join(["---"] * (num_classes + 1)) + "|"
    rows = [header_row, sep_row]

    str_to_frac = lambda s: f"{s.count('O')}/{len(s)}"
    for key, string in d.items():
        row = "|" + key + "|" + "|".join([str_to_frac(s) for s in string]) + "|"
        rows.append(row)
    table = "\n".join(rows)
    return table


def hex_to_rgb(hex_code: str) -> torch.tensor:
    """Converts a hex code to an RGB tensor."""

    hex_code = hex_code.lstrip("#")
    rgb = tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))
    rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
    return rgb_tensor

def plot_accuracy_and_frequency(test_dict:dict, 
                                cfg:SimpleNamespace) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:

    class_acc = pd.DataFrame(
        [
            [
                (
                    sum(1 for i in cl_unit if i == "O") / len(cl_unit)
                    if len(cl_unit) > 1
                    else pd.NA
                )
                for cl_unit in test_dict.get(unit, [])
            ]
            for unit in cfg.names
        ],
        index=cfg.names,
        columns=range(cfg.num_classes),
    )
    class_counts = pd.DataFrame(
        [
            [len(cl_unit) for cl_unit in test_dict.get(unit, [])]
            for unit in cfg.names
        ],
        index=cfg.names,
        columns=range(cfg.num_classes),
    )

    class_counts = class_counts.apply(lambda row: row / row.sum(), axis=1)

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharex=True, sharey=True)
    sns.heatmap(class_acc.fillna(0), ax=ax0, vmin=0, vmax=1)
    sns.heatmap(class_counts, ax=ax1, vmin=0, vmax=1)
    ax0.set_title("Class accuracy")
    ax1.set_title("Class frequency")
    return fig, class_acc, class_counts


def plot_mae_loss(test_dict:dict, cfg:SimpleNamespace) -> tuple[plt.Figure, pd.Series, pd.Series]:
    avg_loss = pd.Series({k: sum(v) / len(v) for k, v in test_dict.items()})
    sample_size = pd.Series({k: len(v) for k, v in test_dict.items()})
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    avg_loss.sort_values(ascending=False).plot.bar(ax=ax0)
    sample_size.sort_values(ascending=False).plot.bar(ax=ax1)
    ax0.set_title("Average MAE loss")
    ax1.set_title("Hours in the test dataset")
    return fig, avg_loss, sample_size