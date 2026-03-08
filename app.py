"""
Streamlit Web App - Bộ Cân Bằng Âm Thanh Số & Phân Loại Thể Loại Nhạc
DSP Final Project - Nhóm ABNQ
"""

import io
import os
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

warnings.filterwarnings('ignore')

# Import core modules
from core.equalizer import (
    BANDS,
    FS,
    apply_equalizer,
    compute_spectrum,
    get_filters,
    load_audio_from_bytes,
)
from core.ml_classifier import (
    extract_features,
    load_classifier,
    predict_genre,
)

# Preset gains (dB) - 10 bands
PRESETS = {
    'Flat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Bass Boost': [6, 4, 2, 0, 0, 0, 0, 0, 0, 0],
    'Treble Boost': [0, 0, 0, 0, 0, 2, 4, 4, 2, 0],
    'Vocal': [0, 0, -2, 2, 4, 4, 2, 0, 0, 0],
    'Rock': [4, 3, 0, 0, 2, 4, 2, 0, 0, 0],
}


def plot_waveform(audio_orig, audio_eq, fs):
    """Vẽ waveform trước và sau equalizer."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t_orig = np.arange(len(audio_orig)) / fs
    t_eq = np.arange(len(audio_eq)) / fs

    axes[0].plot(t_orig, audio_orig, color='steelblue', linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel('Biên độ')
    axes[0].set_title('Waveform — Tín hiệu gốc')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, max(t_orig[-1], t_eq[-1])])

    axes[1].plot(t_eq, audio_eq, color='crimson', linewidth=0.8, alpha=0.9)
    axes[1].set_xlabel('Thời gian (s)')
    axes[1].set_ylabel('Biên độ')
    axes[1].set_title('Waveform — Sau equalizer')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('So sánh Waveform Trước và Sau EQ', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_spectrum(audio_orig, audio_eq, fs, gains_db):
    """Vẽ phổ tần số trước và sau equalizer."""
    freqs_orig, psd_orig = compute_spectrum(audio_orig, fs)
    freqs_eq, psd_eq = compute_spectrum(audio_eq, fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs_orig, psd_orig, color='steelblue', linewidth=1.5, label='Gốc')
    ax.plot(freqs_eq, psd_eq, color='crimson', linewidth=1.5, label='EQ', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlim([20, 22050])
    ax.set_xlabel('Tần số (Hz)')
    ax.set_ylabel('Công suất (dB)')
    ax.set_title('Phổ Tần Số — Gốc vs EQ')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Đánh dấu bands có gain khác 0
    for i, ((name, low, high), gain_db) in enumerate(zip(BANDS, gains_db)):
        if gain_db != 0:
            center = (low * high) ** 0.5
            ax.axvspan(low, high, alpha=0.08, color=plt.cm.tab10(i / len(BANDS)))
            ax.text(center, ax.get_ylim()[0] + 2, f'{gain_db:+.0f}dB',
                    ha='center', fontsize=7, rotation=90)

    plt.tight_layout()
    return fig


def render_equalizer_tab(data):
    """Tab Equalizer. Dùng file đã upload ở sidebar (data = bytes)."""
    st.header('Bộ Cân Bằng Âm Thanh Số (FIR 10-band)')

    # Preset trong sidebar (upload đã chuyển lên main)
    with st.sidebar:
        st.subheader('Cài đặt Equalizer')
        preset_name = st.selectbox(
            'Preset',
            options=list(PRESETS.keys()),
            index=0,
            help='Chọn preset có sẵn (sliders sẽ cập nhật)',
        )

    if not data:
        st.info('👆 Upload file .wav ở sidebar để bắt đầu.')
        return

    # Load audio
    try:
        audio_orig, fs = load_audio_from_bytes(data)
    except Exception as e:
        st.error(f'Không thể đọc file: {e}')
        return

    # Sliders - 2 cột x 5 bands
    st.subheader('Điều chỉnh Gain (dB)')
    base_gains = PRESETS[preset_name]
    gains_db = []

    cols = st.columns(5)
    for i, ((name, low, high), default_gain) in enumerate(zip(BANDS, base_gains)):
        with cols[i % 5]:
            g = st.slider(
                f'{name} ({low}-{high} Hz)',
                min_value=-12,
                max_value=12,
                value=int(default_gain),
                step=1,
            )
            gains_db.append(g)

    # Apply equalizer
    filters = get_filters()
    audio_eq, _ = apply_equalizer(audio_orig, filters, gains_db)

    # Plots
    st.subheader('Waveform')
    fig_wave = plot_waveform(audio_orig, audio_eq, fs)
    st.pyplot(fig_wave)
    plt.close(fig_wave)

    st.subheader('Phổ Tần Số')
    fig_spec = plot_spectrum(audio_orig, audio_eq, fs, gains_db)
    st.pyplot(fig_spec)
    plt.close(fig_spec)

    # Audio player
    st.subheader('Nghe thử')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('Tín hiệu gốc')
        st.audio(data, format='audio/wav')
    with col2:
        st.caption('Sau equalizer')
        buf = io.BytesIO()
        sf.write(buf, audio_eq, fs, format='WAV')
        st.audio(buf.getvalue(), format='audio/wav')

    # Download
    st.download_button(
        label='Tải file đã EQ (.wav)',
        data=buf.getvalue(),
        file_name='audio_equalized.wav',
        mime='audio/wav',
    )


def render_classifier_tab(data):
    """Tab Genre Classifier. Dùng file đã upload ở sidebar (data = bytes)."""
    st.header('Phân Loại Thể Loại Âm Nhạc')

    # Check if model exists
    model_path = 'models/genre_model.joblib'
    le_path = 'models/label_encoder.joblib'
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        st.warning(
            '**Model chưa được train.** Chạy `ml_classification.ipynb` và thực thi cell '
            '"Export Model cho Streamlit App" để tạo file model, sau đó chạy lại app.'
        )
        st.code('''
# Trong ml_classification.ipynb, chạy cell:
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump({"model": best_model, "scaler": scaler}, "models/genre_model.joblib")
joblib.dump(le, "models/label_encoder.joblib")
''', language='python')
        return

    if not data:
        st.info('👆 Upload file .wav ở sidebar để phân loại thể loại nhạc.')
        return

    # Save to temp file (extract_features cần file path)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        pipeline, le = load_classifier()
        model = pipeline['model']
        scaler = pipeline['scaler']

        genre, conf, top3 = predict_genre(tmp_path, model, scaler, le)

        if genre is None:
            st.error('Không thể trích xuất đặc trưng từ file.')
            return

        # Result
        st.success(f'**Dự đoán: {genre.upper()}** (Độ tin cậy: {conf:.1f}%)')

        # Top 3
        st.subheader('Top 3 dự đoán')
        for i, (g, c) in enumerate(top3, 1):
            st.progress(c / 100, text=f'{i}. {g}: {c:.1f}%')

        # Bar chart - full proba
        features = extract_features(tmp_path)
        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            proba = model.predict_proba(features_scaled)[0]
            genre_proba = list(zip(le.classes_, proba * 100))
            genre_proba.sort(key=lambda x: x[1], reverse=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            genres_display = [g for g, _ in genre_proba]
            proba_display = [p for _, p in genre_proba]
            colors = ['gold' if g == genre else 'steelblue' for g in genres_display]
            ax.barh(genres_display, proba_display, color=colors, alpha=0.85)
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Phân bố độ tin cậy theo thể loại')
            ax.axvline(50, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Audio player
        st.audio(data, format='audio/wav')

    except Exception as e:
        st.error(f'Lỗi: {e}')
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    st.set_page_config(
        page_title='DSP Equalizer & Genre Classifier',
        page_icon='🎵',
        layout='wide',
    )
    st.title('🎵 Bộ Cân Bằng Âm Thanh Số & Phân Loại Thể Loại Nhạc')
    st.caption('DSP Final Project — Nhóm ABNQ')

    # Upload file chung ở sidebar — dùng cho cả Equalizer và Phân Loại Thể Loại
    with st.sidebar:
        st.subheader('File âm thanh')
        uploaded = st.file_uploader(
            'Chọn file .wav',
            type=['wav'],
            help='Upload file âm thanh — dùng cho Equalizer và Phân loại thể loại',
        )
        if uploaded:
            st.caption(f'📁 {uploaded.name}')

    # Đọc file 1 lần để dùng chung cho cả 2 tab (tránh read() bị consume)
    data = uploaded.read() if uploaded else None

    tab1, tab2 = st.tabs(['Equalizer', 'Phân Loại Thể Loại'])

    with tab1:
        render_equalizer_tab(data)

    with tab2:
        render_classifier_tab(data)


if __name__ == '__main__':
    main()
