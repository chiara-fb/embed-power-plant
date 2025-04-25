import torch
import torch.nn as nn

#Encoder
class Encoder(nn.Module):
    def __init__(self, rnn_layers, input_feature_len, sequence_len, hidden_size):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.rnn_layers = rnn_layers
        self.rnn_layer = nn.GRU(
            num_layers=rnn_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=rnn_dropout,
            dtype=torch.float64,
        )
        print("Encoder initialized")

    def forward(self, input_seq):

        h0 = torch.zeros(
            self.rnn_layers, input_seq.size(0), self.hidden_size, dtype=torch.float64
        )
        h0 = h0.to(input_seq.device)
        gru_out, hidden = self.rnn_layer(input_seq, h0)
        return gru_out, hidden


# Decoder
class Decoder(nn.Module):
    def __init__(
        self, num_regressors, num_classes, hidden_size, rnn_layers, enc_size
    ):
        
        super().__init__()
        self.num_regressors = num_regressors
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        # self.static_feat_len = static_feat_len
        self.enc_size = enc_size
        # self.dropout = dropout
        self.rnn_layers = rnn_layers

        self.rnn_layer = nn.GRU(
            num_layers=rnn_layers,
            input_size=num_regressors + num_classes,
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=dropout,
            dtype=torch.float64,
        )

        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(
            hidden_size + enc_size, num_classes, bias=False, dtype=torch.float64
        )
        print("Decoder initialized")

    def forward(self, y, prev_hidden, static=None):
        # decoder uses future regressors and hidden states
        output, hidden = self.rnn_layer(y, prev_hidden)
        # if encodings are provided, concatenate them to the input of the output layer
        if static is not None:
            output = torch.cat((output, static.unsqueeze(1)), axis=-1)

        output = self.out_layer(output)

        return output, hidden


class EncoderDecoder(nn.Module):
    """For this EncoderDecoder, only encodings are added after the Decoder step"""
    def __init__(self, cfg):
        super().__init__()

        self.seed = cfg.seed
        self.cfg = cfg
        torch.manual_seed(self.seed)

        for key, val in vars(self.cfg).items():
            setattr(self, key, val)

        self.encoder = Encoder(
            rnn_layers=self.rnn_layers,
            input_feature_len=self.num_regressors + self.num_classes,
            sequence_len=self.in_seq_len,
            hidden_size=self.hidden_size,
            # rnn_dropout=self.enc_dropout,
        )

        self.decoder_cell = Decoder(
            num_regressors=self.num_regressors,
            # out_feat_len=self.num_classes,
            hidden_size=self.hidden_size,
            # dropout=cfg.dec_dropout,
            rnn_layers=self.rnn_layers,
            enc_size=self.enc_size,
            num_classes=self.num_classes,
        )

    def forward(self, x):
        enc_inp, dec_inp, static = x

        # if no encodings are provided, set static to None
        if self.enc_size == 0:
            static = None

        enc_output, enc_hidden = self.encoder(enc_inp)
        # hidden state dim: rnn_layers x batchsize x hidden size
        prev_hidden = enc_hidden
        # uncomment to null the encoder hidden state for testing
        # prev_hidden = torch.zeros(self.rnn_layers, enc_input.size(0), self.hidden_size).float()

        # y_prev = enc_inp[:, -1:, :self.out_feat_len]
        y_prev = enc_inp[:, -1:, : self.num_classes]

        # output dim: batchsize x out_seq_len x forecast features
        # dec_output = torch.zeros(enc_inp.size(0), self.out_seq_len, self.out_feat_len).double()
        dec_output = torch.zeros(
            enc_inp.size(0), self.out_seq_len, self.num_classes
        ).double()
        dec_output = dec_output.to(dec_inp.device)

        # autoregressive prediction loop of decoder
        for i in range(self.out_seq_len):
            step_decoder_input = torch.cat((y_prev, dec_inp[:, [i]]), axis=2)
            rnn_output, prev_hidden = self.decoder_cell(
                step_decoder_input, prev_hidden, static
            )
            y_prev = rnn_output
            dec_output[:, i] = rnn_output.squeeze(1)
        return dec_output

    def to_device(self, device):
        """move layers of all blocks (encoder, decoder) to device"""
        for block in self.children():
            for layer in block.children():
                try:
                    layer = layer.to(device)
                except:
                    print(
                        f"{type(block).__name__} {type(layer).__name__} could not be moved to {device}."
                    )
        return self

    def reset_weights(self):
        """reset layers of all blocks (encoder, decoder)"""
        for block in self.children():
            for layer in block.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
                else:
                    print(
                        f"{type(block).__name__} {type(layer).__name__} has no parameters."
                    )
