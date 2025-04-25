import torch
import torch.nn as nn
from datetime import datetime as dt
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from types import SimpleNamespace
from trainer.train_utils import *
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class Trainer:
    def init_writer(self, writer_dir):
        writer_dir = "//".join(
            [
                "runs",
                "train",
                (
                    writer_dir
                    if writer_dir is not None
                    else dt.today().strftime("%Y%m%d%H%M%S")
                ),
            ]
        )
        self.writer = SummaryWriter(writer_dir)
        self.writer.add_text("Model config", prettify_config(self.model.cfg.__dict__))
        self.writer.add_text("Train config", prettify_config(self.cfg.__dict__))

    def load_data(self, train_dataset, val_dataset=None):
        if self.val_split:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                generator=torch.manual_seed(self.seed),
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=len(val_dataset),
                generator=torch.manual_seed(self.seed),
            )
        else:
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                generator=torch.manual_seed(self.seed),
            )

    def __init__(self, 
                 model:torch.nn.Module, 
                 cfg:SimpleNamespace, 
                 writer_dir:str,
                 train_dataset:Dataset, 
                 val_dataset:Dataset=None):

        self.model = model
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        for key, val in vars(self.cfg).items():
            setattr(self, key, val)

        self.optimizer = getattr(optim, self.optimizer)(
            self.model.parameters(), lr=self.learning_rate
        )
        # Define a StepLR scheduler that decreases the learning rate by a factor of gamma every cfg.lr_step_size
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.lr_step_size, gamma=cfg.gamma
        )
        self.criterion = getattr(nn, self.criterion)()
        self.load_data(train_dataset, val_dataset)
        self.writer_dir = writer_dir
        self.init_writer(writer_dir)

    def iter_step(self, X, Y):
        Y_pred = self.model(X)
        if self.cfg.discretize:
            target = Y.flatten(end_dim=-2).argmax(dim=-1) if Y.ndim > 2 else Y
            output = Y_pred.flatten(end_dim=-2) if Y_pred.ndim > 2 else Y_pred
        else:
            loss = self.criterion(Y_pred, Y)
            return loss, None, None

        loss = self.criterion(output, target)
        acc = accuracy(Y_pred, Y)
        bal_acc = balanced_accuracy(Y_pred, Y)
        return loss, acc, bal_acc  # if self.num_classes else None

    def train_step(self, X, Y):
        self.model.train()
        self.optimizer.zero_grad()
        loss, acc, bal_acc = self.iter_step(X, Y)
        loss.backward()
        self.optimizer.step()
        (
            self.scheduler.step()
            if self.scheduler.get_last_lr()[0] > self.cfg.min_learning_rate
            else None
        )
        return loss, acc, bal_acc

    def eval_step(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            loss, acc, bal_acc = self.iter_step(X, Y)
        return loss, acc, bal_acc

    def early_stopping(self, current_loss, prev_loss):
        self.early_stop_counter = (
            self.early_stop_counter + 1 if current_loss > prev_loss else 0
        )
        stop = True if self.early_stop_counter >= self.early_stop_steps else False
        return stop, current_loss

    def log_metrics(self, metrics_dict, step):
        for name, val in metrics_dict.items():
            self.writer.add_scalar(name, val, step)

    def logout(self):
        self.writer.flush()
        return self.model

    def train(self):
        early_stop, prev_loss, eval_acc, j = 0, torch.inf, torch.inf, 0

        for epoch in range(self.epochs):
            avg_loss, avg_acc, avg_bal_acc = 0, 0, 0
            iterdata = iter(self.train_dataloader)
            train_size = len(iterdata)
            pbar = tqdm(iterable=range(train_size))

            for i in pbar:
                batch = next(iterdata)
                X_batch, Y_batch = (
                    batch[:-1],
                    batch[-1],
                )  # this is needed for compatibility with pbar
                loss, acc, bal_acc = self.train_step(*X_batch, Y_batch)
                avg_loss += loss
                desc = f"Epoch:{epoch} - Avg loss:{(avg_loss/(i+1)):.5f}"
                if self.cfg.discretize:
                    avg_acc += acc
                    avg_bal_acc += bal_acc
                    desc += f" - Avg acc:{(avg_acc/(i+1)):.5f} - Avg bal acc:{(avg_bal_acc/(i+1)):.5f}"

                if self.val_split and i % self.val_steps == 0 and i != 0:
                    j += 1
                    val_data = next(iter(self.val_dataloader))
                    X_val, Y_val = val_data[:-1], val_data[-1]
                    eval_loss, eval_acc, eval_bal_acc = self.eval_step(*X_val, Y_val)
                    self.log_metrics(
                        {"Eval loss": eval_loss}, j
                    )  # , 'Eval accuracy': eval_acc}, j)
                    early_stop, prev_loss = self.early_stopping(eval_loss, prev_loss)
                    if early_stop:
                        self.logout()

                pbar.set_description(desc)
                self.log_metrics({"Loss": loss}, i + train_size * epoch)
                (
                    self.log_metrics(
                        {"Accuracy": acc, "Balanced accuracy": bal_acc},
                        i + train_size * epoch,
                    )
                    if self.cfg.discretize
                    else None
                )
            #should only add embeddings if the model contains them
            if "Embed" in str(type(self.model)):
                emb_layer = (
                    self.model.encoder.emb_layer.weight.detach().T
                    if "EmbedEncoderDecoder" in str(type(self.model))
                    else self.model.decoder_cell.emb_layer.weight.detach().T
                )

                self.writer.add_embedding(
                    emb_layer,
                    metadata=self.cfg.units,
                    global_step=epoch,
                    label_img=torch.stack(
                        [torch.tensor([1.0, 0.0, 0.0]) for unit in self.cfg.units]
                    )
                    .view(-1, 3, 1, 1)
                    .repeat(1, 1, 10, 10),
                )

            self.log_metrics({"Avg epoch loss": avg_loss / train_size}, epoch)
            (
                self.log_metrics({"Avg epoch accuracy": avg_acc / train_size,
                                  "Avg epoch bal acc": avg_bal_acc/train_size}, epoch)
                if self.cfg.discretize
                else None
            )
            self.logout()

    def test_classes(self, test_dataloader):
        test_dict = {}
        for i, (X_test, Y_test) in enumerate(test_dataloader):
            with torch.no_grad():
                Y_pred = self.model(X_test)
            
            # get unit the sample belongs to
            unit = self.model.cfg.units[X_test[-1].argmax()]
            test_dict[unit] = test_dict.get(unit, [""] * self.model.cfg.num_classes)

            for i in range(self.model.cfg.out_seq_len):
                acc = accuracy(Y_pred[:, i], Y_test[:, i])
                test_dict[unit][Y_test[:, i].argmax()] += "O" if acc else "X"

        self.writer.add_text(
        "Test", prettify_test(test_dict, num_classes=self.model.cfg.num_classes) )
        fig, class_acc, class_counts = plot_accuracy_and_frequency(test_dict, self.cfg)
        self.writer.add_figure("Test frequency and accuracy", fig)

        return class_acc, class_counts

    def test_regressor(self, test_dataloader):
        test_dict = {}
        for i, (X_test, Y_test) in enumerate(test_dataloader):
            Y_pred = self.model(X_test)
            # get unit the sample belongs to
            unit = self.model.cfg.units[X_test[-1].argmax()]
            criterion = nn.L1Loss(reduction="none")
            loss = criterion(Y_pred, Y_test).detach()
            test_dict[unit] = test_dict.get(unit, []) + loss.flatten().tolist()

        fig, avg_loss, sample_size = plot_mae_loss(test_dict, self.cfg)    
        self.writer.add_text("Test", avg_loss.to_markdown())
        self.writer.add_figure("Loss and reporting hours", fig)

        return avg_loss, sample_size

    def test(self, test_dataset):
        single_batch_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        full_batch_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
       
        self.model.eval()
        
        X_test, Y_test = next(iter(full_batch_loader))
        test_loss, test_acc, test_bal_acc = self.eval_step(X_test, Y_test)
        self.log_metrics({"Test loss": test_loss}, 0)

        if self.discretize:
            self.log_metrics(
                {
                    "Test accuracy": test_acc,
                    "Test balanced accuracy": test_bal_acc,
                },
                0,
            )
        
            return self.test_classes(single_batch_loader)
        else:
            test_loss, test_acc, test_bal_acc = self.eval_step(X_test, Y_test)
            return self.test_regressor(single_batch_loader)
        

