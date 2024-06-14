from dataclasses import dataclass
import json

import numpy as np
from transformers.modeling_outputs import ModelOutput
import torch
from torch import nn

# from omegaconf import OmegaConf, DictConfig

class RNN_Layer(torch.nn.Module):
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


@dataclass
class VMDModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class VMDModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = OmegaConf.create(config) # config with omegaconf
        self.num_classes = config["num_classes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.p_rnn_layer = RNN_Layer(
            input_size=config["phonetic_encoder"]["input_size"],
            hidden_size=config["phonetic_encoder"]["hidden_size"],
            dropout=config["phonetic_encoder"]["drop_out"],
        )

        self.l_embs_vocab = nn.Embedding(
            self.num_classes, config["linguistic_encoder"]["embs_dim"]
        )

        self.l_rnn_layer = RNN_Layer(
            input_size=config["linguistic_encoder"]["embs_dim"],
            hidden_size=config["linguistic_encoder"]["hidden_size"],
            dropout=config["linguistic_encoder"]["drop_out"],
        )

        self.mha_decoder = torch.nn.MultiheadAttention(
            embed_dim=config["linguistic_encoder"]["hidden_size"] * 2,
            num_heads=config["decoder"]["mha"]["num_heads"],
            dropout=config["decoder"]["mha"]["dropout"],
        )

        self.feed_foward_decoder = nn.Sequential(
            nn.BatchNorm1d(config["linguistic_encoder"]["hidden_size"] * 4),
            nn.ReLU(),
            nn.Dropout(p=config["decoder"]["feed_forward"]["dropout"]),
            nn.Linear(
                config["linguistic_encoder"]["hidden_size"] * 4, self.num_classes
            ),
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, phonetic, canonical, transcript=None, input_sizes=None, label_sizes=None
    ):
        query = self.p_rnn_layer(phonetic)
        query = phonetic.transpose(0, 1).contiguous() + query

        # Linguistic
        can_embs = self.l_embs_vocab(canonical)
        can_tensor = self.l_rnn_layer(can_embs)

        # Decoder
        mha_output, _ = self.mha_decoder(query, can_tensor, can_tensor)

        output = torch.concat((query, mha_output), dim=-1)

        L, N, _ = output.size()
        output = output.view(L * N, -1)
        output = self.feed_foward_decoder(output)
        output = output.view(L, N, -1)
        output = self.log_softmax(output)

        if transcript is not None:
            loss_fn = nn.CTCLoss(reduction="sum")
            loss = loss_fn(output, transcript, input_sizes, label_sizes)
            loss /= N
        else:
            loss = 0

        return VMDModelOutput(logits=output, loss=loss)

    def get_predict(self, predicts, predict_sizes):
        list_predict = []
        for i in range(len(predicts)):
            pred = self._post_process(predicts[i], predict_sizes[i])
            list_predict.append(pred)

        return list_predict

    def _post_process(self, predict, input_size=None):
        pred = []
        predict_actual = predict[:input_size] if input_size is not None else predict[:]
        for i in range(len(predict_actual)):
            if predict[i] == 0:
                continue
            if predict[i] == 1 and input_size is not None:
                continue
            if i == 0:
                pred.append(predict[i])
            if i > 0 and predict[i] != predict[i - 1]:
                if len(pred) == 0:
                    pred.append(predict[i])
                elif predict[i] != pred[-1]:
                    pred.append(predict[i])
        return pred

    def get_param_size(self):
        """
        In ra số lượng tham số  của mô hình và kích thước, trọng lượng của mô hình (MB)
        """
        totaconfig = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_size = sum(p.numel() * p.element_size() for p in self.parameters())

        total_size_mb = total_size / (1024 * 1024)
        print(
            f"Số lượng tham số : {round(totaconfig/1000000, 3)}M . Kích thước mô hình : {round(total_size_mb, 3)} MB"
        )

    def predict(self, model, vocab, device, phonetic_embs, canonical_phoneme):
        canonical_phoneme = canonical_phoneme.split()
        canonical_phoneme = torch.IntTensor(
            [*map(vocab["label2index"].get, canonical_phoneme)]
        ).unsqueeze(0)
        with torch.no_grad():
            output = model(phonetic_embs.to(device), canonical_phoneme.to(device))
            _, predicts = torch.max(output.logits, dim=-1)
            predicts = predicts.transpose(0, 1).cpu().numpy()
        transcript = self._post_process(predicts[0])
        transcript = [str(element) for element in transcript]
        transcript = [*map(vocab["index2label"].get, transcript)]
        return transcript
