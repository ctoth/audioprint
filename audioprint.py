import sys
import audioread
import numpy as np
import librosa
import scipy.fftpack
from typing import Tuple


def read_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    with audioread.audio_open(file_path) as audio_file:
        sr = audio_file.samplerate
        num_channels = audio_file.channels

        raw_pcm_data = bytearray()

        # Read and concatenate audio frames
        for frame in audio_file:
            raw_pcm_data.extend(frame)

        # Convert the byte array to a numpy array
        pcm_data = np.frombuffer(raw_pcm_data, dtype=np.int16)

        # If the audio has more than one channel, average the channels to obtain a mono signal
        if num_channels > 1:
            pcm_data = pcm_data.reshape((-1, num_channels)).mean(axis=1)

        return pcm_data, sr


def audio_phash(raw_pcm_data: np.ndarray, sr: int, n_mfcc: int = 32) -> int:
    S = librosa.feature.melspectrogram(y=raw_pcm_data, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)

    # Compute the DCT of the mean of MFCC matrix
    dct_coefficients = scipy.fftpack.dct(np.mean(mfcc, axis=1), norm='ortho')

    # Compute the median value of DCT coefficients
    median = np.median(dct_coefficients)

    # Generate a binary hash by comparing DCT coefficients with the median value
    bit_array = (dct_coefficients > median).astype(int)

    # Convert the binary hash to an integer
    hash_value = int("".join(map(str, bit_array)), 2)

    return hash_value


def fingerprint_file(file_path: str, n_mfcc: int = 32) -> int:
    raw_pcm_data, sr = read_audio_file(file_path)
    return audio_phash(raw_pcm_data, sr, n_mfcc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audioprint.py <path/to/audio/file>")
        sys.exit(1)

    file_path = sys.argv[1]
    hash_value = fingerprint_file(file_path)
    print(f"Perceptual hash for the audio file '{file_path}': {hash_value}")
