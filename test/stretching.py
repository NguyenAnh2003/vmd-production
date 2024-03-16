import numpy as np
from scipy.io import wavfile
from time_stretch import *

# Đọc file Wav
SAMPLE_RATE, sample = wavfile.read("data_generate/data_original/test.wav")

print(SAMPLE_RATE)
print(sample.shape)

dtype = sample.dtype
# sample = torch.tensor(
#     [np.swapaxes(sample, 0, 1)],  # (samples, channels) --> (channels, samples)
#     dtype=torch.float32,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# )


def test_time_stretch_2_up():
    # Tăng tốc x1.25
    up = time_stretch(sample, 1.25, SAMPLE_RATE)
    wavfile.write(
        "./data_generate/data_stretch/stretch_up_2.wav",
        SAMPLE_RATE,
        np.swapaxes(up.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


def test_time_stretch_2_down():
    # Làm chậm x0.75
    down = time_stretch(sample, 0.75, SAMPLE_RATE)
    wavfile.write(
        "./data_generate/data_stretch/stretch_down_2.wav",
        SAMPLE_RATE,
        np.swapaxes(down.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


# Hàm dùng để test tốc độ audio nếu như không biết chính xác tốc độ đang cần
# def test_time_stretch_to_fast_ratios():
#     # có được tỷ lệ kéo dài nhanh (tốc độ từ 50% đến 200%)
#     for ratio in get_fast_stretches(SAMPLE_RATE):
#         print("Stretching", ratio)
#         stretched = time_stretch(sample, ratio, SAMPLE_RATE)
#         wavfile.write(
#             f"./data_generate/stretched_ratio_{ratio.numerator}-{ratio.denominator}.wav",
#             SAMPLE_RATE,
#             np.swapaxes(stretched.cpu()[0].numpy(), 0, 1).astype(dtype),
#         )

# test_time_stretch_2_up()
