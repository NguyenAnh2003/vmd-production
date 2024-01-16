import torch
import torchaudio
from utils.dataset.data_loader import Vocab
from phonetic_embedding import phonetic_embedding
import edit_distance
from model import customize_model


def test_1_audio(Model_Training, path_save_model, audio, canonical):
    try:
        device = torch.device('cpu')
        vocab = Vocab("../utils/dataset/phoneme.txt")
        new_path_save_model = path_save_model

        package = torch.load(new_path_save_model, map_location=device)

        a_param = package["a_param"]
        pi_param = package["pi_param"]
        p_param = package["p_param"]
        l_param = package["l_param"]

        model = Model_Training(a_param, pi_param, p_param, l_param, vocab)

        model.load_state_dict(package['state_dict'], strict=True)
        model.to(device)
        model.eval()

        audio_array, _ = torchaudio.load(audio)
        audio_array = audio_array.squeeze(0)
        phonetic = phonetic_embedding(audio_array).unsqueeze(0)

        phonetic = phonetic.to(device)

        canonical = canonical.split()
        canonical = torch.IntTensor([*map(vocab.word2index.get, canonical)]).unsqueeze(0)
        canonical = canonical.to(device)

        with torch.no_grad():
            inputs = (None, None, phonetic), canonical
            output = model(inputs)

            _, predicts = torch.max(output, dim=-1)

            predict = predicts.transpose(0, 1).cpu().numpy()[0]
            predict = edit_distance.preprocess_predict(predict, len(predict))
            predict_phoneme = [*map(vocab.index2word.get, predict)]

        return predict_phoneme
    except Exception as e:
        print(f"{e}")
        raise e


if __name__ == "__main__":
    a = test_1_audio(Model_Training=customize_model.Model,
                     path_save_model="../saved_model/model_Customize_All_3e3.pth",
                     audio="vao-nui_1618754929889.wav",
                     canonical="v aː-1 uz $ n w i-4")
    print(a)
