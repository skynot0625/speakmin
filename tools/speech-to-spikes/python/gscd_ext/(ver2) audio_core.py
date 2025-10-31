import matplotlib.pyplot as plt
import librosa
import math
from pydub import AudioSegment
from scipy import interpolate
import numpy as np
from pydub.utils import (audioop, ratio_to_db, get_array_type)
import array
import gc
from scipy import signal

class AudioCore(object):
    def __init__(self, file, format='wav', ch_index=0):
        assert ch_index in [0, 1], f'ch_index({ch_index}) not supported'
        try:
            self.wav_file = file
            self.sound = AudioSegment.from_file(file, format=format)
            self.channels = self.sound.channels
            self.frame_rate = self.sound.frame_rate
            self.duration = self.sound.duration_seconds
            self.sample_width = self.sound.sample_width
            self.dBFS = self.sound.dBFS
            self.dt = 1.0 / self.frame_rate
            
            # AGC 파라미터 초기화
            self.agc_fast_gain = 1.0
            self.agc_slow_gain = 1.0
            self.target_level = 0.5
            self.min_gain = 0.05
            self.max_gain = 20.0
            
            self._create_sig()
            self._create_time()
            self._do_fft()

            self.filter_banks = {}
        except Exception as e:
            print(f"Error initializing AudioCore for {file}: {str(e)}")
            raise

    def _create_sig(self):
        try:
            if self.channels == 1:
                self.sig = np.array(self.sound.get_array_of_samples(), dtype='float', copy=True)
            elif self.channels == 2:
                self.sig = np.array(self.sound.get_array_of_samples(), dtype='float', copy=True)[ch_index::2]
            del self.sound
            gc.collect()
        finally:
            gc.collect()

    def _create_time(self):
        self.n_points = len(self.sig)
        self.time_s = 0.0
        self.time_e = self.duration
        self.time = np.linspace(self.time_s, self.time_e, self.n_points, endpoint=False)

    def _do_fft(self):
        window = np.hanning(self.n_points)
        self.dft_x = np.fft.fft(self.sig * window)
        del window
        gc.collect()

    def apply_agc(self, signal, tau_fast=0.001, tau_slow=0.1):
        """실시간 AGC 적용"""
        try:
            rms = np.sqrt(np.mean(signal**2))
            target_gain = self.target_level / (rms + 1e-6)
            
            # 빠른 AGC
            self.agc_fast_gain = np.clip(target_gain, self.min_gain, self.max_gain)
            
            # 느린 AGC
            self.agc_slow_gain += (self.agc_fast_gain - self.agc_slow_gain) * tau_slow
            self.agc_slow_gain = np.clip(self.agc_slow_gain, self.min_gain, self.max_gain)
            
            return signal * self.agc_slow_gain
        finally:
            gc.collect()

    def create_mel_filterbank(self, n_mels):
        """필터뱅크를 한 번만 생성"""
        if n_mels in self.filter_banks:
            return self.filter_banks[n_mels]
            
        nyq = self.frame_rate / 2
        min_freq = 20
        max_freq = min(nyq * 0.95, 8000)
        
        positions = np.linspace(0, 1, n_mels)
        freqs = 165.4 * (10**(2.1 * positions) - 0.88)
        freqs = np.clip(freqs, min_freq, max_freq)
        
        # 필터 계수 미리 계산
        filters = []
        for freq in freqs:
            low = max(0.001, min(0.98, (freq * 0.95) / nyq))
            high = max(0.02, min(0.99, (freq * 1.05) / nyq))
            b, a = signal.butter(2, [low, high], btype='band')
            filters.append((b, a))
        
        self.filter_banks[n_mels] = (freqs, filters)
        return freqs, filters

    def process_audio_chunk(self, n_mels, chunk_size=1024):
        _, filters = self.create_mel_filterbank(n_mels)
        
        for i in range(0, len(self.sig), chunk_size):
            chunk = self.sig[i:i + chunk_size]
            chunk_filtered = []
            
            for b, a in filters:
                filtered = signal.filtfilt(b, a, chunk)
                normalized_filtered = self.apply_agc(filtered)
                chunk_filtered.append(normalized_filtered)
            
            yield chunk_filtered


    def speech2spikes_realtime(self, n_mels, vth, **kwargs):
        try:
            all_spikes = []
            neuron_potentials = [0] * n_mels  # 각 채널별 뉴런 전위 저장
            
            for filtered_chunks in self.process_audio_chunk(n_mels):
                chunk_spikes = []
                for ch, chunk in enumerate(filtered_chunks):
                    potential_t, potential_v, time_spike = self.integrate_and_fire(
                        chunk, vth, initial_potential=neuron_potentials[ch], **kwargs
                    )
                    neuron_potentials[ch] = potential_v[-1] if potential_v else 0
                    chunk_spikes.extend([[t, ch] for t in time_spike])
                
                all_spikes.extend(sorted(chunk_spikes, key=lambda x: x[0]))
            
            return all_spikes
        finally:
            gc.collect()

    def integrate_and_fire(self, vin, vth, alpha=1, leak_enable=False, leak_tau=16000e-6, initial_potential=0):
        time_spike_list = []
        potential_t = []
        potential_v = []
        potential = initial_potential
        
        for index, v in enumerate(vin):
            t = self.time[index]

            if index == 0:
                potential_t.append(t)
                potential_v.append(potential)
                s = 0
                t_pre = t
                v_pre = v
                continue

            if v * v_pre < 0:
                t_0 = t_pre + abs(v_pre / v) * (t - t_pre)
                s = abs((t_0 - t_pre) * v_pre / 2) + abs((t - t_0) * v / 2)
            else:
                s = abs(v_pre + v) * (t - t_pre) / 2

            potential_pre = potential
            if leak_enable:
                potential = potential * math.exp(- (t - t_pre) / leak_tau) + s * alpha
            else:
                potential += s * alpha

            if vth <= potential:
                while vth <= potential:
                    time_spike = t_pre + (t - t_pre) * (vth - potential_pre) / (potential - potential_pre)
                    potential_t.append(time_spike)
                    potential_v.append(vth)
                    potential_next = potential - vth
                    potential_pre = potential
                    potential = 0
                    potential_t.append(time_spike)
                    potential_v.append(potential)
                    potential_pre = potential
                    potential = potential_next
                    if potential < vth:
                        potential_t.append(t)
                        potential_v.append(potential)
                    time_spike_list.append(time_spike)
            else:
                potential_t.append(t)
                potential_v.append(potential)

            v_pre = v
            t_pre = t

        return potential_t, potential_v, time_spike_list

    def plot_mel_filter_banks(self, ax, n_mels, norm='slaney', echo=True):
        melfb_2d = librosa.filters.mel(sr=self.frame_rate, n_fft=self.n_points, n_mels=n_mels, norm=norm)
        melfb_freq = librosa.fft_frequencies(sr=self.frame_rate, n_fft=self.n_points)
        for index, melfb in enumerate(melfb_2d):
            ax.plot(melfb_freq, melfb)
            if echo:
                print(f'ch={index}, freq={melfb_freq[np.argmax(melfb)]}, {melfb[np.argmax(melfb)]}')

    def plot_dft(self, ax):
        freq = self.get_fft_freq()
        ax.plot(freq, np.real(self.dft_x))
        del freq

    def plot_original(self, ax):
        ax.plot(self.time, self.sig)

    def get_fft_freq(self):
        return np.fft.fftfreq(self.n_points, self.dt)
