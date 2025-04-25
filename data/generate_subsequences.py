import torch
from torch.utils.data import Dataset
from data.data_utils import *
from types import SimpleNamespace

class GenDataset(Dataset):
    """Base class to generate a Dataset from 
    a tuple of X,y tensors."""
    def __init__(
        self, X,y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def train_val_test_split(        
    market_data: torch.tensor,
    units_data: torch.tensor,
    unit_encodings: torch.tensor,
    cfg: SimpleNamespace,
    device:torch.device = None) -> tuple[Dataset, Dataset, Dataset]:
    """Split the data into train, validation and test sets.
    """

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []
    _, n_units = units_data.shape

    ### Move data to device
    market_data = market_data.to(device=device)
    units_data = units_data.to(device=device)
    unit_encodings = unit_encodings.to(device=device)

    ### Loop over units
    for unit in range(n_units):
        unit_time_series = units_data[:, unit]
        unit_time_series = strip_zeros(unit_time_series)

        #have to discretize it here, because the strip_zeros would 
        # otherwise remove trailing zeros in the discretization
        if cfg.discretize:
            unit_time_series = discretize(unit_time_series, cfg.num_classes)
        else:
            unit_time_series = unit_time_series.unsqueeze(-1)

        n_steps = len(unit_time_series)
        n_sequences = (n_steps - (cfg.in_seq_len + cfg.out_seq_len)) // cfg.in_step_size + 1

        perm_ix = torch.randperm(n_sequences, generator=torch.manual_seed(cfg.seed))
        val_size = int(cfg.val_split * n_sequences)
        test_size = int(cfg.test_split * n_sequences)
        train_ix, val_ix, test_ix = (perm_ix[:(-(val_size+test_size) 
                                               if val_size+test_size > 0 else None)], 
                                        perm_ix[(val_size+test_size):-test_size], 
                                        perm_ix[-test_size:])
        
        assert n_sequences == len(train_ix) + val_size + test_size, "Splitting error"
        assert len(range(0, n_steps - cfg.in_seq_len - cfg.out_seq_len + 1, cfg.in_step_size)) == n_sequences, "n_sequences error"

        ### generate subsequences of fixed-length for each unit
        for ix, seq_start in enumerate(range(
            0,
            n_steps - (cfg.in_seq_len + cfg.out_seq_len) + 1,
            cfg.in_step_size,
            )):

            seq_end = seq_start + cfg.in_seq_len
            x = torch.cat(
                (unit_time_series[seq_start:seq_end, :], market_data[seq_start:seq_end, :]),
                dim=1, 
            )
            xf = market_data[seq_end : seq_end + cfg.out_seq_len, :]
            tr = unit_time_series[seq_end : seq_end + cfg.out_seq_len]
            st = unit_encodings[unit]

            ### assign the subsequence to train, val or test
            if ix in train_ix:
                X_train.append((x, xf, st))
                Y_train.append(tr)
            elif ix in val_ix:
                X_val.append((x, xf, st))
                Y_val.append(tr)
            else:
                X_test.append((x, xf, st))
                Y_test.append(tr)

            ### assertion to check correctness of the iteration
            assert len(tr) == cfg.out_seq_len, "Added incomplete samples"

    #transform to torch.utils.data.Dataset
    train_dataset = GenDataset(X_train, Y_train)
    val_dataset = GenDataset(X_val, Y_val)
    test_dataset = GenDataset(X_test, Y_test)

    return train_dataset, val_dataset, test_dataset



