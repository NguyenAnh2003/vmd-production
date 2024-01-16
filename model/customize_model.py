import torch
import torch.nn as nn
from model.rnn_stack import RNN_Layer
from model.vmd_model import Model_Main
from utils.dataset.data_loader import Vocab
import argparse
import yaml

def load_config(path: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default=path)
    args = parser.parse_args()
    config_path = args.conf
    param = yaml.safe_load(open(config_path, 'r', encoding="utf8"))
    return param

class Model(Model_Main):
    def __init__(self, a_param=None, pi_param=None, p_param=None, l_param=None, vocab=None):
        """
        cnn_param [dict]:  cnn parameters, only support Conv2d i.e.
        rnn_param [dict]:  rnn parameters i.e.
        vocab_size  [int]:  Sizes of vocab, default: 54
        """
        super().__init__(a_param, pi_param, p_param, l_param, vocab)
        p_param['cnn_layers'] = 0
        p_param['rnn_layers'] = 1
        self.p_cnn_stack, self.p_rnn_stack = self._encoder(p_param)

        l_param["embs_type_dim"] = 256
        self.l_embs1 = nn.Embedding(vocab.n_words, l_param["embs_dim"])  # Phoneme Embedding
        self.l_embs2 = nn.Embedding(6, l_param["embs_type_dim"])  # Phoneme Embedding

        l_param["rnn_hidden_size"] = 384

        self.l_lstm = RNN_Layer(input_size=l_param["embs_dim"]+l_param["embs_type_dim"],
                                hidden_size=l_param["rnn_hidden_size"],
                                bidirectional=l_param["rnn_bidirectional"],
                                batch_first=l_param["rnn_batch_first"],
                                dropout=0.2)


        self.decoder = torch.nn.MultiheadAttention(embed_dim=l_param["rnn_hidden_size"] * 2,
                                                   num_heads=4,
                                                   dropout=0.1)


        self.decoder_fc = nn.Sequential(nn.BatchNorm1d(l_param["rnn_hidden_size"] * 2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(l_param["rnn_hidden_size"]*2, vocab.n_words))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        (_, _, ph), can = x
        ph = self._forward_encoder(ph, self.p_cnn_stack, self.p_rnn_stack)

        query = ph.transpose(0, 1).contiguous()

        # Linguistic
        can_embs = self.l_embs1(can)

        can_index_type = [[*map(self.vocab.index_type_phoneme.get, seq)] for seq in can.cpu().numpy()]
        can_index_type = torch.IntTensor(can_index_type)
        can_embs_type = self.l_embs2(can_index_type)

        can_embs = torch.concat((can_embs, can_embs_type), dim=-1)

        can_tensor = self.l_lstm(can_embs)
        can_tensor = can_tensor.transpose(0, 1).contiguous()

        output, _ = self.decoder(query, can_tensor, can_tensor)

        output = torch.add(output, query)

        L, N, _ = output.size()
        output = output.view(L * N, -1)
        output = self.decoder_fc(output)
        output = output.view(L, N, -1)

        return self.log_softmax(output)

if __name__ == '__main__':
    # a_param = load_config("../config/acoustic_param.yaml")
    # pi_param = load_config("../config/pitch_param.yaml")
    p_param = load_config("../configs/phonetic_param.yaml")
    l_param = load_config("../configs/linguistic_param.yaml")
    vocab = Vocab("../utils/dataset/phoneme.txt")
    vocab = Vocab("../utils/dataset/phoneme.txt")
    model = Model(a_param=None, pi_param=None, p_param=p_param, l_param=l_param, vocab=vocab).cuda()

    for idx, m in enumerate(model.children()):
        print("{} - {}".format(idx, m))
    print("Number of parameters %d" % sum([param.nelement() for param in model.parameters()]))

    acoustic = torch.randn(32, 100, 81).cuda()
    pitch = torch.randn(32, 200, 1).cuda()
    phonetic = torch.randn(32, 99, 768).cuda()
    canonical = torch.randint(low=0, high=54, size=(32, 100)).cuda()

    a = model(((acoustic, pitch, phonetic), canonical))
    print(a.shape)
