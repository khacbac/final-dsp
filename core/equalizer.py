"""
Core equalizer logic - FIR 10-band graphic equalizer.
Extracted from equalizer.ipynb for use by Streamlit app.
"""

import io
import numpy as np
import scipy.signal as signal
import soundfile as sf

# =============================================================
# THÔNG SỐ EQUALIZER
# =============================================================
FS = 44100          # Tần số lấy mẫu (Hz)
NUMTAPS = 1025      # Số tap FIR (phải lẻ cho linear phase)

# 10 bands: (tên, tần số thấp, tần số cao) — đơn vị Hz
BANDS = [
    ('Sub-bass',   20,    60),
    ('Bass',       60,    170),
    ('Low-mid',    170,   310),
    ('Mid',        310,   600),
    ('Upper-mid',  600,   1000),
    ('Presence',   1000,  3000),
    ('Brilliance', 3000,  6000),
    ('Air',        6000,  10000),
    ('High air',   10000, 16000),
    ('Ultra-high', 16000, 20000),
]


def design_fir_bandpass(low_hz, high_hz, fs=FS, numtaps=NUMTAPS):
    """
    Thiết kế bộ lọc FIR bandpass bằng phương pháp cửa sổ Hamming.

    Tham số:
        low_hz  : tần số cắt thấp (Hz)
        high_hz : tần số cắt cao (Hz)
        fs      : tần số lấy mẫu (Hz)
        numtaps : số hệ số FIR (phải lẻ)

    Trả về:
        b : mảng hệ số FIR, shape=(numtaps,)
    """
    nyquist = fs / 2.0

    # Chuẩn hóa về [0, 1] so với Nyquist
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist

    # Giới hạn để tránh lỗi
    low_norm = max(low_norm, 1e-4)
    high_norm = min(high_norm, 1.0 - 1e-4)

    b = signal.firwin(
        numtaps,
        [low_norm, high_norm],
        pass_zero=False,     # Bandpass (không đi qua DC)
        window='hamming'
    )
    return b


def get_filters(bands=BANDS, fs=FS, numtaps=NUMTAPS):
    """Trả về list filters cho 10 bands."""
    return [design_fir_bandpass(low, high, fs, numtaps) for _, low, high in bands]


def load_audio(file_path, target_fs=FS, silent=False):
    """
    Load file audio, resample về target_fs nếu cần, chuyển về mono.

    Trả về:
        audio : numpy array float32, shape=(N,)
        fs    : tần số lấy mẫu thực tế
    """
    audio, fs = sf.read(file_path, dtype='float32')

    # Chuyển stereo → mono (lấy trung bình 2 kênh)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
        if not silent:
            print('Stereo → Mono (trung bình 2 kênh)')

    # Resample nếu cần
    if fs != target_fs:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        if not silent:
            print(f'Resampled: {fs} Hz → {target_fs} Hz')

    # Chuẩn hóa về [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio, fs


def load_audio_from_bytes(data_bytes, target_fs=FS):
    """
    Load audio từ bytes (dùng cho file upload trong Streamlit).

    Tham số:
        data_bytes : bytes của file .wav
        target_fs  : tần số lấy mẫu mong muốn

    Trả về:
        audio : numpy array float32, shape=(N,)
        fs    : tần số lấy mẫu thực tế
    """
    buf = io.BytesIO(data_bytes)
    audio, fs = sf.read(buf, dtype='float32')

    # Chuyển stereo → mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample nếu cần
    if fs != target_fs:
        import librosa
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    # Chuẩn hóa về [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio, fs


def apply_equalizer(audio, filters, gains_db):
    """
    Áp dụng bộ lọc FIR equalizer.

    Quy trình:
    1. Với mỗi band: lọc tín hiệu qua FIR bandpass
    2. Nhân với gain (linear)
    3. Cộng tất cả bands lại
    4. Chuẩn hóa để tránh clip
    """
    output = np.zeros_like(audio)
    band_signals = []

    for band_filter, gain_db in zip(filters, gains_db):
        filtered = signal.lfilter(band_filter, [1.0], audio)
        gain_linear = 10 ** (gain_db / 20.0)
        band_out = filtered * gain_linear
        band_signals.append(band_out)
        output += band_out

    # Chuẩn hóa để tránh clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val

    return output, band_signals


def compute_spectrum(audio, fs, nperseg=4096):
    """
    Tính phổ công suất trung bình (Power Spectral Density) dùng Welch method.
    """
    freqs, psd = signal.welch(
        audio, fs=fs, nperseg=nperseg,
        window='hamming', scaling='spectrum'
    )
    psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
    return freqs, psd_db
