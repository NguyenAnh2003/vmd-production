import torch
from collections import OrderedDict


class RNN_Layer(torch.nn.Module):
    """
    1 lớp RNN sẽ bao gồm LSTM, batchnorm, dropout
    Shape input có dạng (batch_size, n_frames, d_model)
    Args:
        input_size: Kích thước input size (là kích thước d_model)
        hidden_size: Kích thước lớp ẩn (cũng là lớp đầu ra), Nếu bidirectional True thì output_size = 2*hidden_size
        num_batch_norm: đầu vào của batch_norm, giá trị của nó là n_frames
        bidirectional=True: True thì trở thành Bi-LSTM
        batch_first=False: True thì shape x là (batch_size, n_frames, d_model), False thì (n_frames, batch_size, d_model)
        dropout=0.1: Tỉ lệ bỏ học
    """

    def __init__(self, input_size, hidden_size, bidirectional=True, batch_first=False, dropout=0.1, is_batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                                  batch_first=batch_first)
        self.is_batch_norm = is_batch_norm
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size*2) if is_batch_norm else None
        self.layer_norm = torch.nn.LayerNorm(hidden_size*2) if not is_batch_norm else None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): Có Kích thước (batch_size, n_frames, d_model)
        Returns:
            Tensor: nn.Bi-LSTM -> nn.BatchNorm -> nn.Dropout
        """

        # Nếu batch_first là True thì không cần thay đổi shape của x
        if self.batch_first:
            x, _ = self.lstm(x)
            if self.is_batch_norm:
                x = x.transpose(1, 2).contiguous()
                x = self.batch_norm(x)
                x = x.transpose(1, 2).contiguous()
            else:
                x = self.layer_norm(x)
        # Nếu batch_first là False thì cần thay đổi shape x thành (n_frames, batch_size, d_model)
        else:
            # (batch_size, n_framse, d_model) -> (n_frames, batch_size, d_model)
            x = x.transpose(0, 1).contiguous()
            x, _ = self.lstm(x)

            # (n_frames, batch_size, d_model) -> (batch_size, n_frames, d_model) dể batch_norm
            x = x.transpose(0, 1).contiguous()
            if self.is_batch_norm:
                x = x.transpose(1, 2).contiguous()
                x = self.batch_norm(x)
                x = x.transpose(1, 2).contiguous()
            else:
                x = self.layer_norm(x)

        x = self.dropout(x)
        return x

class RNN_Stack(torch.nn.Module):
    """
        n_layers lớp RNN_Layer
        Shape input có dạng (batch_size, n_frames, d_model)
        Args:
            n_layers -> int: Số lớp RNN_Layer
            parameters -> list: Tham số của RNN_Layer gồm 4 keys :input_size, hidden_size, bidirectional, batch_first, được lưu dưới dạng dict
                        VD: parameters = {
                                    "input_size" : 128,
                                    "hidden_size" : 128,
                                    "bidirectional" : True,
                                    "batch_first" : False
                                }
            dropout -> float: tỉ lệ bỏ học
        """

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters

        n_layers = parameters["layers"]
        dropout = parameters["drop_out"]
        input_size = parameters["input_size"]
        hidden_size = parameters["hidden_size"]
        bidirectional = parameters["bidirectional"]
        num_directions = 2 if bidirectional else 1
        batch_first = parameters["batch_first"]
        device = eval(parameters["device"])
        dtype = eval(parameters["dtype"])

        stack = []
        rnn_layer = RNN_Layer(input_size=input_size,
                              hidden_size=hidden_size,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout,
                              device=device,
                              dtype=dtype)
        stack.append(('0', rnn_layer))
        for n in range(n_layers - 1):
            rnn_layer = RNN_Layer(input_size=num_directions * hidden_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  batch_first=batch_first,
                                  dropout=dropout,
                                  device=device,
                                  dtype=dtype)

            stack.append(('%d' % (n + 1), rnn_layer))
        self.rnn_stack = torch.nn.Sequential(OrderedDict(stack))

    def forward(self, x):
        return self.rnn_stack(x)
