import numpy as np
import librosa
import scipy.fftpack

import audioread


def read_audio_file(file_path):
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


def audio_phash(raw_pcm_data, sr, n_mfcc=64):
    # Compute the spectrogram
    S = np.abs(librosa.stft(raw_pcm_data))

    # Convert the spectrogram to the Mel scale
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr)

    # Compute the MFCCs
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_S), n_mfcc=n_mfcc)

    # Average the MFCCs across time
    mfcc_mean = np.mean(mfcc, axis=1)

    # Apply the DCT
    dct_coeffs = scipy.fftpack.dct(mfcc_mean, norm='ortho')[:n_mfcc]

    # Compute the hash
    median_val = np.median(dct_coeffs)
    hash_val = ''.join('1' if coeff > median_val else '0' for coeff in dct_coeffs)

    # Convert the binary hash to an integer
    int_hash = int(hash_val, 2)

    return int_hash

import sys
import audioread
import numpy as np
import librosa
import scipy.fftpack

# Include the read_audio_file and audio_phash functions here

def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_phash_script.py <path/to/audio/file>")
        sys.exit(1)

    file_path = sys.argv[1]
    hash_value = fingerprint_file(file_path)

    print(f"Perceptual hash for the audio file '{file_path}': {hash_value}")

def fingerprint_file(file_path):
    raw_pcm_data, sr = read_audio_file(file_path)
    hash_value = audio_phash(raw_pcm_data, sr)
    return hash_value

if __name__ == "__main__":
    main()
