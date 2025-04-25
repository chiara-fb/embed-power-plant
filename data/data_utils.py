import torch
from torch.nn.functional import one_hot
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

def discretize(tensor: torch.tensor, num_classes: int) -> torch.tensor:
    """Discretize the output tensor to a one-hot tensor with num_classes classes.
    Parameters:
        tensor: tensor of shape [batch_size, out_seq_len, num_classes]
        num_classes: int indicating the number of classes to discretize the tensor into
    """
    tensor = tensor * num_classes
    tensor = tensor.clip(0, num_classes - 1)
    tensor = one_hot(tensor.to(torch.int64), num_classes)
    return tensor.to(torch.float64)


def strip_zeros(tensor: torch.tensor) -> torch.tensor:
    """strip leading and trailing zeros from a torch.tensor.

    Parameters:
        series[torch.tensor]: the generation series
    Output:
        series[pd.Series]: the generation series with leading and trailing zeros removed (length may vary)
    """

    cum_pos = torch.cumsum(tensor > 0, dim=0)
    if cum_pos.max() == 0:
        return torch.tensor([])

    first_leading_pos = torch.nonzero(cum_pos, as_tuple=False)[0]
    first_trailing_zero = cum_pos.argmax() + 1
    return tensor[first_leading_pos:first_trailing_zero]


def map_encoding(word: str, mapping: list, others_cat: bool = True) -> torch.tensor:
    """Given a word and a list of patterns, finds the pattern
    that return a one-hot tensor encoding the word.
    Parameters:
        word: str to encode
        mapping: list of regex patterns to match
        others_cat: whether to add an "others" category if no match is found
    """
    dim = len(mapping)
    for i, el in enumerate(mapping):
        if re.search(el, word):
            return one_hot(torch.tensor(i), dim + int(others_cat))

    if others_cat:
        return one_hot(torch.tensor(dim), dim + int(others_cat))
    else:
        return torch.zeros(dim, dtype=torch.float64)


def map_encodings(word: str, mappings: list, others_cat: bool = True) -> torch.tensor:
    encodings = [map_encoding(word, m, others_cat=others_cat) for m in mappings]
    return torch.cat(encodings, dim=-1).to(torch.float64)


def open_mod_to_excel():
    """Scrape the Open Models wiki page and save the table to an Excel file."""
    #Fetch the HTML
    url = "https://wiki.openmod-initiative.org/wiki/Open_Models"  
    response = requests.get(url)
    html_content = response.content
    #Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    #Locate the table
    table = soup.find('table', {'class': 'sortable wikitable smwtable'})  # Adjust selector as needed
    #Extract the header
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())
    #Extract table rows
    rows = []
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    #Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)
    df.to_excel('open_models.xlsx', index=False)
    return df