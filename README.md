# Audioprint

Audioprint is a Python library that computes a perceptual hash (or fingerprint) for audio files.
The goal of Audioprint is to generate similar hash values for similar-sounding audio files, making it suitable for tasks such as audio comparison, deduplication, or identification.
The library processes audio files in various formats (based on the available backends) and outputs an integer hash value.

## Concept

Audioprint works by transforming audio data into a perceptually meaningful representation that captures the spectral characteristics of the audio file. This is achieved through a series of steps, including:

* Converting the audio data to a time-frequency representation using a spectrogram.
* Transforming the linear frequency scale of the spectrogram into the Mel scale, which approximates human auditory perception.
* Extracting Mel-frequency cepstral coefficients (MFCCs) from the Mel-scaled spectrogram, which capture the spectral shape of the sound.
* Reducing the dimensionality of the MFCC matrix by computing the average of each MFCC across time and applying the discrete cosine transform (DCT).
* Generating the hash by comparing the DCT coefficients to their median value and converting the binary representation to an integer.

## Dependencies

* Python 3.6 or later
* audioread
* numpy
* scipy
* librosa

To install the required packages using the provided ```requirements.txt``` file, run:

```bash
pip install -r requirements.txt
```

## UsageAs a library

To use Audioprint in your Python projects, first, import the required functions:

```python
from audioprint import fingerprint_file
```

Then, compute the perceptual hash for the specified audio file:

```python
file_path = 'path/to/your/audio_file.ext'
hash_value = fingerprint_file(file_path)
```

## Functions

Audioprint provides a main function:
```python
fingerprint_file(file_path: str, n_mfcc: int = 32) -> int
```
:
Reads an audio file, computes the perceptual hash for the given file, and returns an integer hash value.

Additionally, the library exposes two lower-level functions:

```python
read_audio_file(file_path: str) -> Tuple[np.ndarray, int]
```
:
Reads an audio file using ```audioread``` and returns the raw PCM data and sample rate.

```python
audio_phash(raw_pcm_data: np.ndarray, sr: int, n_mfcc: int = 32) -> int
```
:
Computes the perceptual hash for the given raw PCM data and sample rate, returning an integer hash value.

## Contributing

Feel free to submit pull requests or open issues to report bugs, request new features, or suggest improvements.

## License

This project is licensed under the MIT License. See the ```LICENSE``` file for more details.
