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
from streamlit_vertical_slider import vertical_slider

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


@st.cache_resource
def get_filters_cached():
    """Cache FIR filters - computed once, never change."""
    return get_filters()


@st.cache_resource
def load_classifier_cached():
    """Cache ML model loading."""
    return load_classifier()


def plot_waveform(audio_orig, audio_eq, fs):
    """Vẽ waveform trước và sau equalizer."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor('#1e293b')
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#f1f5f9')
        ax.yaxis.label.set_color('#f1f5f9')
        ax.title.set_color('#f1f5f9')
    t_orig = np.arange(len(audio_orig)) / fs
    t_eq = np.arange(len(audio_eq)) / fs

    axes[0].plot(t_orig, audio_orig, color='#818cf8', linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel('Biên độ')
    axes[0].set_title('Waveform — Tín hiệu gốc')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, max(t_orig[-1], t_eq[-1])])

    axes[1].plot(t_eq, audio_eq, color='#f472b6', linewidth=0.8, alpha=0.9)
    axes[1].set_xlabel('Thời gian (s)')
    axes[1].set_ylabel('Biên độ')
    axes[1].set_title('Waveform — Sau equalizer')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('So sánh Waveform Trước và Sau EQ', fontsize=14, fontweight='bold', y=1.02, color='#f1f5f9')
    plt.tight_layout()
    return fig


def plot_spectrum(audio_orig, audio_eq, fs, gains_db):
    """Vẽ phổ tần số trước và sau equalizer."""
    freqs_orig, psd_orig = compute_spectrum(audio_orig, fs)
    freqs_eq, psd_eq = compute_spectrum(audio_eq, fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#f1f5f9')
    ax.yaxis.label.set_color('#f1f5f9')
    ax.title.set_color('#f1f5f9')
    ax.plot(freqs_orig, psd_orig, color='#818cf8', linewidth=1.5, label='Gốc')
    ax.plot(freqs_eq, psd_eq, color='#f472b6', linewidth=1.5, label='EQ', alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlim([20, 22050])
    ax.set_xlabel('Tần số (Hz)')
    ax.set_ylabel('Công suất (dB)')
    ax.set_title('Phổ Tần Số — Gốc vs EQ')
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f1f5f9')
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


def render_genre_block(data):
    """Compact genre classification block - only when file uploaded."""
    model_path = 'models/genre_model.joblib'
    le_path = 'models/label_encoder.joblib'
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        return

    if not data:
        return

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with st.spinner('Đang phân loại thể loại...'):
            pipeline, le = load_classifier_cached()
            model = pipeline['model']
            scaler = pipeline['scaler']
            genre, conf, top3 = predict_genre(tmp_path, model, scaler, le)

        if genre is None:
            return

        # Compact genre block - Option A/B hybrid
        top3_str = ' | '.join(f'{g}: {c:.1f}%' for g, c in top3)
        with st.container():
            st.success(f'**Genre: {genre.upper()}** ({conf:.1f}%)')
            st.caption(top3_str)

        # Full distribution in expander
        with st.expander('Phân bố đầy đủ theo thể loại'):
            features = extract_features(tmp_path)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                proba = model.predict_proba(features_scaled)[0]
                genre_proba = list(zip(le.classes_, proba * 100))
                genre_proba.sort(key=lambda x: x[1], reverse=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#1e293b')
                ax.tick_params(colors='#94a3b8')
                ax.xaxis.label.set_color('#f1f5f9')
                ax.yaxis.label.set_color('#f1f5f9')
                ax.title.set_color('#f1f5f9')
                genres_display = [g for g, _ in genre_proba]
                proba_display = [p for _, p in genre_proba]
                colors = ['#fbbf24' if g == genre else '#818cf8' for g in genres_display]
                ax.barh(genres_display, proba_display, color=colors, alpha=0.85)
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Phân bố độ tin cậy theo thể loại')
                ax.axvline(50, color='#f87171', linestyle='--', alpha=0.5)
                ax.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    except Exception as e:
        st.error(f'Lỗi phân loại: {e}')
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def render_equalizer(data):
    """Main equalizer section."""
    st.header('Bộ Cân Bằng Âm Thanh Số (FIR 10-band)')

    if not data:
        st.info('👆 Upload file .wav ở sidebar để bắt đầu.')
        return

    # Load audio
    try:
        with st.spinner('Đang tải audio...'):
            audio_orig, fs = load_audio_from_bytes(data)
    except Exception as e:
        st.error(f'Không thể đọc file: {e}')
        return

    # Preset from sidebar (already rendered)
    preset_name = st.session_state.get('preset_select', 'Flat')
    base_gains = PRESETS[preset_name]

    # Reset sliders when preset changes or Flatten All clicked
    if 'preset_prev' not in st.session_state:
        st.session_state['preset_prev'] = preset_name
    if st.session_state['preset_prev'] != preset_name:
        for i in range(10):
            st.session_state.pop(f'eq_band_{i}', None)
        st.session_state['preset_prev'] = preset_name

    # Presets + Flatten All + Bypass row
    st.subheader('Điều chỉnh Gain (dB)')
    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
    with btn_col1:
        if st.button('Flatten All', help='Reset tất cả bands về 0 dB'):
            for i in range(10):
                st.session_state.pop(f'eq_band_{i}', None)
            st.session_state['preset_select'] = 'Flat'
            st.session_state['preset_prev'] = 'Flat'
            st.rerun()
    with btn_col3:
        bypass_eq = st.checkbox('Bypass EQ', key='bypass_eq', help='Bỏ qua equalizer, output = input')

    # Vertical sliders - 10 bands left-to-right
    cols = st.columns(10)
    gains_db = []
    for i, ((name, low, high), default_gain) in enumerate(zip(BANDS, base_gains)):
        with cols[i]:
            g = vertical_slider(
                label=f'{name}',
                key=f'eq_band_{i}',
                min_value=-12,
                max_value=12,
                default_value=int(default_gain),
                step=1,
                height=180,
                value_always_visible=True,
            )
            gains_db.append(g)
            st.caption(f'{low}-{high} Hz')

    # Apply equalizer (or bypass)
    if bypass_eq:
        audio_eq = audio_orig.copy()
    else:
        filters = get_filters_cached()
        with st.spinner('Đang áp dụng equalizer...'):
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
        buf_value = buf.getvalue()
        st.audio(buf_value, format='audio/wav')

    # Download
    st.download_button(
        label='Tải file đã EQ (.wav)',
        data=buf_value,
        file_name='audio_equalized.wav',
        mime='audio/wav',
    )


def main():
    st.set_page_config(
        page_title='DSP Equalizer & Genre Classifier',
        page_icon='🎵',
        layout='wide',
    )

    # Inject custom CSS
    css_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'static', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title('🎵 Bộ Cân Bằng Âm Thanh Số & Phân Loại Thể Loại Nhạc')
    st.caption('DSP Final Project — Nhóm ABNQ')

    # Sidebar: File upload + Preset
    with st.sidebar:
        st.subheader('File âm thanh')
        uploaded = st.file_uploader(
            'Chọn file .wav',
            type=['wav'],
            help='Upload file âm thanh — dùng cho Equalizer và Phân loại thể loại',
        )
        if uploaded:
            size_kb = getattr(uploaded, 'size', 0) / 1024
            st.caption(f'📁 {uploaded.name} ({size_kb:.1f} KB)')

        st.subheader('Cài đặt Equalizer')
        preset_name = st.selectbox(
            'Preset',
            options=list(PRESETS.keys()),
            index=0,
            help='Chọn preset có sẵn (sliders sẽ cập nhật)',
            key='preset_select',
        )
        st.caption('---')
        st.caption('DSP Final Project — Nhóm ABNQ')

    # Read file once for both genre and equalizer
    data = uploaded.read() if uploaded else None

    if not uploaded:
        st.session_state.pop('last_uploaded_name', None)

    # Toast on new file upload
    if data and uploaded:
        prev_name = st.session_state.get('last_uploaded_name')
        if prev_name != uploaded.name:
            st.session_state['last_uploaded_name'] = uploaded.name
            st.toast(f'Đã tải lên: {uploaded.name}', icon='✅')

    # Genre block (compact) - only when file uploaded
    if data:
        render_genre_block(data)
        st.divider()

    # Equalizer (main)
    render_equalizer(data)


if __name__ == '__main__':
    main()
