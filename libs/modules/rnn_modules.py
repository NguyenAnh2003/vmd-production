import torch

class RNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=False,
        )
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size * 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        x, _ = self.lstm(x)
        # (n_frames, batch_size, d_model) ->  (batch_size, d_model, n_frames) dể batch_norm
        x = x.permute(1, 2, 0).contiguous()
        x = self.batch_norm(x)
        x = x.permute(2, 0, 1).contiguous()  # (n_frames, batch_size, d_model)

        x = self.dropout(x)
        return x
