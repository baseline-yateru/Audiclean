import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np



class AudioProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_data, self.sample_rate = librosa.load(file_path, mono=False, sr=None)
        self.fft_rate = 2048  # Default FFT rate

    @property
    def channel_count(self):
        return 1 if self.audio_data.ndim == 1 else self.audio_data.shape[0]

    def set_fft_rate(self, fft_rate):
        self.fft_rate = fft_rate
    def get_duration(self):
        return librosa.get_duration(y=self.audio_data, sr=self.sample_rate)

    def get_sample_rate(self):
        return self.sample_rate

    def get_spectrum(self):
        return np.stack(
            [
                librosa.amplitude_to_db(np.abs(librosa.stft(channel, n_fft=self.fft_rate)))
                for channel in self._channels()
            ]
        )


    def plot_waveform(self):
        fig, axes = plt.subplots(self.channel_count, 1, figsize=(14, 4 * self.channel_count), squeeze=False)
        for index, channel in enumerate(self._channels()):
            librosa.display.waveshow(channel, sr=self.sample_rate, ax=axes[index, 0])
            axes[index, 0].set_title(f"Waveform - channel {index + 1}")
            axes[index, 0].set_xlabel("Time (s)")
            axes[index, 0].set_ylabel("Amplitude")
        fig.tight_layout()
        plt.show()

    def plot_spectrogram(self):
        fig, axes = plt.subplots(self.channel_count, 1, figsize=(14, 4 * self.channel_count), squeeze=False)
        for index, channel in enumerate(self._channels()):
            spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(channel)))
            image = librosa.display.specshow(
                spectrum,
                sr=self.sample_rate,
                x_axis="time",
                y_axis="hz",
                ax=axes[index, 0],
            )
            axes[index, 0].set_title(f"Spectrogram - channel {index + 1}")
            fig.colorbar(image, ax=axes[index, 0], format="%+2.0f dB")
        fig.tight_layout()
        plt.show()

    def get_mfcc(self, n_mfcc=13):
        return np.stack(
            [
                librosa.feature.mfcc(y=channel, sr=self.sample_rate, n_mfcc=n_mfcc)
                for channel in self._channels()
            ]
        )

    def _channels(self):
        return self.audio_data if self.audio_data.ndim == 2 else (self.audio_data,)

