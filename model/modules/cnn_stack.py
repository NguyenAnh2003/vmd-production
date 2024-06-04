import torch.nn as nn
from collections import OrderedDict

class CNN_Layer(nn.Module):
    """
    1 lớp CNN sẽ bao gồm Conv2d, batchnorm, activation và dropout
    Shape input có dạng (batch_size, n_channel, n_frames, d_model)
    Args:
        in_channel -> int: Số channel đầu vào
        out_channel -> int: Số channel đầu ra
        kernel_size -> int hoặc tuple: Kích thước của kernel
        stride -> int hoặc tuple: Sải bước
        padding -> int hoặc tuple: Đệm các đường biên
        dropout -> float: tỉ lệ bỏ học
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        @param x: Shape input có dạng (batch_size, in_channel, n_frames, d_model)
        @return: Shape input có dạng (batch_size, out_channel, n_frames_new, d_model_new)
                 Conv2d -> Batch_Norm2d -> ReLu -> Dropout
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CNN_Stack(nn.Module):
    """
        n_layers lớp CNN_Layer
        Shape input có dạng (batch_size, n_channel, n_frames, d_model)
        Args:
            n_layers -> int: Số lớp CNN_Layer
            parameters -> list: Tham số của CNN_Layer, có hình dạng (n_layers, 4)
                    Hình dạng các phần tử (in_channel, out_channel), kernel_size, stride, padding
                        [[(1,32), 3, 1, 0], [(32,32), 3, 1, 0]]
            dropout -> float: tỉ lệ bỏ học
        """

    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters
        n_layers = parameters["layers"]
        drop_out = parameters["drop_out"]
        device = eval(parameters["device"])
        dtype = eval(parameters["dtype"])

        assert n_layers <= len(parameters), "Missing parameter for CNN_Stack"

        stack = []
        for n in range(n_layers):
            in_channel = eval(parameters["channel"])[n][0]
            out_channel = eval(parameters["channel"])[n][1]
            kernel_size = eval(parameters["kernel_size"])[n]
            stride = eval(parameters["stride"])[n]
            padding = eval(parameters["padding"])[n]

            cnn_layer = CNN_Layer(in_channel, out_channel, kernel_size, stride, padding, dropout=drop_out,
                                  device=device, dtype=dtype)
            stack.append(('%d' % n, cnn_layer))
        self.cnn_stack = nn.Sequential(OrderedDict(stack))

    def forward(self, x):
        return self.cnn_stack(x)
