import torch
import torchaudio
import io
from utils.translate import translate, get_vocab_from_file
from utils.constants import TEXT2PHONEME
import model.edit_distance as edit_distance
from feats.phonetic_embedding import phonetic_embedding
from utils.dataset.data_loader import Vocab
from model.customize_model import Model
from model.metric import Align

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
            # audio_array = audio_array.reshape(1, audio_array.shape[0]*audio_array.shape[1]).squeeze(0)
            audio_array = torch.mean(audio_array, dim=0)
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

            result = " ".join(predict_phoneme)

            return result

        """ define canonical """
        canonical = translate(sentence=text,
                               method="text_to_phoneme",
                               my_vocab=my_vocab)  # translate 2 phoneme

        """ prediction call function """
        predicted = prediction(Model_Training=Model,
                            path_save_model="./saved_model/model_Customize_All_3e3.pth",
                            audio=media, canonical=canonical)

        canonical = " ".join(canonical.split())
        canonical = canonical.split()
        list_compare = compare_transcript_canonical(canonical, predicted)
        compared = displace_word_mispronounce(canonical, list_compare)
        target_list = text.split() # split text by white space and form list
        result = dict(zip(target_list, compared)) # mapping 2 list and form dict
        return result
    except Exception as e:
        print(f"Error at service class: {e}")
        raise e

def compare_transcript_canonical(canonical, transcript):
    result = []
    can_align, trans_align = Align(canonical, transcript.split())
    for can, tran in zip(can_align, trans_align):
        if can != "<eps>" and tran == "<eps>":
            result.append(1)
        elif can != "<eps>" and tran != "<eps>":
            if can != tran:
                result.append(1)
            else:
                result.append(0)
    return result

def displace_word_mispronounce(canonicals, list_compare):
    phoneme_each_word_compare = []
    phoneme_one_word_compare = []
    for i, (can, compare) in enumerate(zip(canonicals, list_compare)):

        if can != "$":
            phoneme_one_word_compare.append(compare)
        if can == "$" or i == len(canonicals)-1:
            phoneme_each_word_compare.append(phoneme_one_word_compare)
            phoneme_one_word_compare = []
    list_result = [0 if any(list_one_word) else 1 for list_one_word in phoneme_each_word_compare]
    return list_result

if __name__ == "__main__":
    media = "/home/vkuai/Downloads/VMD_VLSP/private_test/audio_data/private_test/cao-gay_1618758834960.wav"
    text = "cao đen gầy" # Cần try catch lại
    print(correcting_service(media, text))