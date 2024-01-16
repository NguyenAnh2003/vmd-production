import torch
import torchaudio
import io

# service class
def correcting_service(media, text):
    """ the service responsible for dealing with wav file
    :return the output
    """
    audio, _ = torchaudio.load(io.BytesIO(media))
    print(audio)
    return text, audio