import torch
import torchaudio
import io
from utils.translate import translate, get_vocab_from_file
from utils.constants import TEXT2PHONEME
import model.edit_distance as edit_distance
from feats.phonetic_embedding import phonetic_embedding
from utils.dataset.data_loader import Vocab
from model.customize_model import Model

# service class
def correcting_service(media, text):
    """ the service responsible for dealing with wav file
    translate text to phoneme and pass the audio to prediction function
    :return the output
    """
    try:
        """ get vocab """
        my_vocab = get_vocab_from_file("./utils/dataset/lexicon_vmd.txt")

        """ prediction function """
        def prediction(Model_Training, path_save_model, audio, canonical):
            device = torch.device('cpu')
            vocab = Vocab("./utils/dataset/phoneme.txt")
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

        # """ prediction call function """
        # result = prediction(Model_Training=Model,
        #            path_save_model="../saved_model/model_Customize_All_3e3.pth",
        #            audio=media)

        result = translate(sentence=text,
                           method="text_to_phoneme",
                           my_vocab=my_vocab) # translate 2 phoneme
        print(f"Translate text 2 phoneme: {result}")

        return f"got result"
    except Exception as e:
        print(f"Error at service class: {e}")
        raise e