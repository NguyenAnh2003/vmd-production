import torch
import torchaudio
import io
from utils.translate import translate, get_vocab_from_file
from utils.constants import TEXT2PHONEME

# service class
def correcting_service(media, text):
    """ the service responsible for dealing with wav file
    :return the output
    """
    audio, _ = torchaudio.load(io.BytesIO(media))
    phoneme = translate(text, TEXT2PHONEME)
    print(audio, phoneme)
    return text, audio