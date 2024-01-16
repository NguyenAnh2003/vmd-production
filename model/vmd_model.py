import math
import torch
import torch.nn as nn
from collections import OrderedDict
from model.cnn_stack import CNN_Layer
from model.rnn_stack import RNN_Layer
from model.metric import Accuracy

class Model_Main(nn.Module):
    def __init__(self, a_param=None, pi_param=None, p_param=None, l_param=None, vocab=None):
        """
        cnn_param [dict]:  cnn parameters, only support Conv2d i.e.
        rnn_param [dict]:  rnn parameters i.e.
        vocab_size  [int]:  Sizes of vocab, default: 54
        """
        super().__init__()
        self.vocab = vocab
        self.a_param = a_param
        self.pi_param = pi_param
        self.p_param = p_param
        self.l_param = l_param

    def _encoder(self, param=None):
        rnn_input_size = param["rnn_input_size"]
        if param["cnn_layers"] != 0:
            # CNN Stack
            cnn_drop_out = param["cnn_drop_out"]

            stack = []
            for n in range(param["cnn_layers"]):
                in_channel = eval(param["cnn_channel"])[n][0]
                out_channel = eval(param["cnn_channel"])[n][1]
                kernel_size = eval(param["cnn_kernel_size"])[n]
                stride = eval(param["cnn_stride"])[n]
                padding = eval(param["cnn_padding"])[n]

                cnn_layer = CNN_Layer(in_channel, out_channel, kernel_size, stride, padding, dropout=cnn_drop_out)
                stack.append(('%d' % n, cnn_layer))

                rnn_input_size = int(math.floor((rnn_input_size + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)
            cnn_stack = nn.Sequential(OrderedDict(stack))
            rnn_input_size = rnn_input_size * out_channel if out_channel else rnn_input_size
        else:
            cnn_stack = None
        # RNN Stack
        rnn_dropout = param["rnn_drop_out"]
        rnn_hidden_size = param["rnn_hidden_size"]
        rnn_bidirectional = param["rnn_bidirectional"]
        rnn_num_directions = 2 if rnn_bidirectional else 1
        rnn_batch_first = param["rnn_batch_first"]

        stack = []
        rnn_layer = RNN_Layer(input_size=rnn_input_size,
                              hidden_size=rnn_hidden_size,
                              bidirectional=rnn_bidirectional,
                              batch_first=rnn_batch_first,
                              dropout=rnn_dropout)
        stack.append(('0', rnn_layer))
        for n in range(param["rnn_layers"] - 1):
            rnn_layer = RNN_Layer(input_size=rnn_num_directions * rnn_hidden_size,
                                  hidden_size=rnn_hidden_size,
                                  bidirectional=rnn_bidirectional,
                                  batch_first=rnn_batch_first,
                                  dropout=rnn_dropout)

            stack.append(('%d' % (n + 1), rnn_layer))
        rnn_stack = torch.nn.Sequential(OrderedDict(stack))
        return cnn_stack, rnn_stack

    def _decoder(self, query, key, value):
        attn_score = query @ key.transpose(1, 2)
        attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True)
        exp_score = torch.exp(attn_score - attn_max)

        attn_weights = exp_score
        weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)
        attn_weights = attn_weights / (weights_denom + 1e-30)
        context = attn_weights @ value

        # query and context concat
        x = torch.concat((query, context), dim=-1)
        # x = torch.add(query, context)
        x = x.transpose(0, 1).contiguous()

        return x

    def _forward_encoder(self, x, cnn_stack, rnn_stack):
        if cnn_stack:
            x = cnn_stack(x.unsqueeze(1))
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = rnn_stack(x)
        return x

    def _padding_to_concat(self, ac=None, pi=None, ph=None):
        ac_frames = ac.shape[1]if torch.is_tensor(ac) else 0
        pi_frames = pi.shape[1] if torch.is_tensor(pi) else 0
        ph_frames = ph.shape[1] if torch.is_tensor(ph) else 0
        max_n_frames = max(ac_frames, pi_frames, ph_frames)
        if ac_frames < max_n_frames if torch.is_tensor(ac) else 0:
            zeros = torch.zeros(ac.shape[0], max_n_frames - ac_frames, ac.shape[2]).cuda()
            ac = torch.concat((ac, zeros), dim=1)
        if pi_frames < max_n_frames and torch.is_tensor(pi):
            zeros = torch.zeros(pi.shape[0], max_n_frames - pi_frames, pi.shape[2]).cuda()
            pi = torch.concat((pi, zeros), dim=1)
        if ph_frames < max_n_frames if torch.is_tensor(ph) else 0:
            zeros = torch.zeros(ph.shape[0], max_n_frames - ph_frames, ph.shape[2]).cuda()
            ph = torch.concat((ph, zeros), dim=1)
        return ac, pi, ph

    def compute_wer(self, index, input_sizes, targets, target_sizes):
        batch_errs = 0
        batch_tokens = 0
        for i in range(len(index)):
            label = targets[i][:target_sizes[i]]
            label = list(filter(lambda x: x != 1, label))
            pred = self.preprocess_predict(index[i], input_sizes[i])
            # for j in range(len(index[i][:input_sizes[i]])):
            #     if index[i][j] == 0 or index[i][j] == 1:
            #         continue
            #     if j == 0:
            #         pred.append(index[i][j])
            #     if j > 0 and index[i][j] != index[i][j - 1]:
            #         if len(pred) == 0:
            #             pred.append(index[i][j])
            #         elif index[i][j] != pred[-1]:
            #             pred.append(index[i][j])

            err, len_ = Accuracy(label, pred)
            batch_errs += err
            batch_tokens += len_

        return batch_errs, batch_tokens

    def preprocess_predict(self, predict, input_size):
        pred = []
        for i in range(len(predict[:input_size])):
            if predict[i] == 0 or predict[i] == 1:
                continue
            if i == 0:
                pred.append(predict[i])
            if i > 0 and predict[i] != predict[i - 1]:
                if len(pred) == 0:
                    pred.append(predict[i])
                elif predict[i] != pred[-1]:
                    pred.append(predict[i])
        return pred

    @staticmethod
    def save_package(model, optimizer=None, paramaters=None, list_acc_loss=None):
        package = {
            'a_param': model.a_param,
            'pi_param': model.pi_param,
            'p_param': model.p_param,
            'l_param': model.l_param,
            'vocab': model.vocab,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if paramaters is not None:
            package['paramaters'] = paramaters
        if list_acc_loss is not None:
            list_train_acc, list_train_loss, list_dev_acc, list_dev_loss = list_acc_loss
            package['list_train_acc'] = list_train_acc
            package['list_train_loss'] = list_train_loss
            package['list_dev_acc'] = list_dev_acc
            package['list_dev_loss'] = list_dev_loss

        return package
