"""
Core ML classifier logic - Genre classification from audio features.
Extracted from ml_classification.ipynb for use by Streamlit app.
"""

import os
import numpy as np
import librosa

# Tham số feature extraction (phải khớp với notebook)
SAMPLE_RATE = 22050   # Hz (librosa default)
DURATION = 30         # giây (GTZAN chuẩn)
N_MFCC = 20          # Số hệ số MFCC
N_CHROMA = 12         # Số bins Chroma
N_CONTRAST = 7       # Số bands Spectral Contrast

GENRES = sorted([
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
])


def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Trích xuất feature vector từ file audio.

    Tham số:
        file_path : đường dẫn file .wav
        sr        : tần số lấy mẫu (Hz)
        duration  : thời lượng tối đa (giây)

    Trả về:
        features : numpy array shape=(82,) hoặc None nếu lỗi
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)

        features = []

        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # 2. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # 3. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_bands=N_CONTRAST - 1
        )
        features.extend(np.mean(contrast, axis=1))
        features.extend(np.std(contrast, axis=1))

        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 5. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f'Lỗi extract_features: {file_path} — {e}')
        return None


def predict_genre(file_path, model, scaler, label_encoder):
    """
    Dự đoán thể loại của 1 file audio.

    Trả về:
        predicted_genre : tên thể loại dự đoán
        confidence      : độ tin cậy (%)
        top3            : list of (genre, confidence) cho top 3
    """
    features = extract_features(file_path)
    if features is None:
        return None, None, None

    features_scaled = scaler.transform(features.reshape(1, -1))

    pred_idx = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    predicted_genre = label_encoder.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx] * 100

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        (label_encoder.inverse_transform([i])[0], float(proba[i] * 100))
        for i in top3_idx
    ]

    return predicted_genre, confidence, top3


def load_classifier(
    model_path='models/genre_model.joblib',
    le_path='models/label_encoder.joblib',
):
    """
    Load model và label encoder từ file.

    Trả về:
        pipeline : dict với keys 'model', 'scaler'
        label_encoder : sklearn LabelEncoder
    """
    import joblib

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        raise FileNotFoundError(
            f"Model chưa được train. Chạy ml_classification.ipynb và thực thi "
            f"cell 'Export Model' để tạo {model_path} và {le_path}"
        )

    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    return pipeline, label_encoder
