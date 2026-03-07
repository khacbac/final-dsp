# DSP Final Project — FIR Equalizer & Music Genre Classifier

**Môn:** Xử Lý Tín Hiệu Số | **Nhóm:** [Nhóm X]

## Tổng Quan

Dự án gồm 2 module chạy trong Jupyter Notebook, và **Streamlit Web App** để tương tác:

| Module | File | Mô tả |
|--------|------|-------|
| FIR Equalizer | `equalizer.ipynb` | Bộ cân bằng âm thanh 10 bands, FIR Hamming |
| ML Classifier | `ml_classification.ipynb` | Phân loại 10 thể loại nhạc (SVM + Random Forest) |
| **Streamlit App** | `app.py` | GUI web: Equalizer + Phân loại thể loại |

---

## Cài Đặt

### Yêu cầu
- Python **3.10+**
- pip

### Cài thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install numpy scipy librosa soundfile matplotlib seaborn scikit-learn ipywidgets jupyter streamlit joblib
```

### Kiểm tra cài đặt

```bash
python -c "import numpy, scipy, librosa, soundfile, matplotlib, sklearn; print('OK')"
```

---

## Cách Chạy

### 1. Khởi động Jupyter

```bash
jupyter notebook
```

### 2. Chạy Equalizer

Mở `equalizer.ipynb`:
- Đổi `AUDIO_FILE` ở cell 5 sang file `.wav` của bạn
- Điều chỉnh `GAINS_DB` ở cell 6 (đơn vị dB, từ -12 đến +12)
- Chạy: **Kernel → Restart & Run All**

```
Đầu ra:
  audio_equalized.wav       — File âm thanh đã cân bằng
  frequency_response.png    — Đáp ứng tần số 10 bands
  waveform_comparison.png   — Waveform trước/sau EQ
  spectrum_comparison.png   — Phổ tần số trước/sau EQ
  band_energy.png           — Năng lượng từng band
```

### 3. Chạy Streamlit Web App (GUI)

```bash
streamlit run app.py
```

Mở trình duyệt tại http://localhost:8501

**Tab Equalizer:** Upload file .wav → điều chỉnh 10 sliders → xem waveform/spectrum → tải file đã EQ.

**Tab Phân Loại Thể Loại:** Upload file .wav → xem dự đoán thể loại + độ tin cậy.

> **Lưu ý:** Tab Phân Loại cần model đã train. Chạy `ml_classification.ipynb` và thực thi cell "Export Model cho Streamlit App" để tạo `models/genre_model.joblib` và `models/label_encoder.joblib`.

### 4. Chạy ML Classifier

Mở `ml_classification.ipynb`:
- Đảm bảo thư mục `train-data/` và `test-data/` tồn tại
- Đổi `PREDICT_FILE` ở cell 14 để dự đoán file mới
- Chạy: **Kernel → Restart & Run All** (~10–20 phút lần đầu)

```
Đầu ra:
  data_distribution.png     — Phân phối dataset
  mel_spectrograms.png      — Mel Spectrogram 4 genres
  model_comparison.png      — So sánh SVM vs Random Forest
  confusion_matrix.png      — Confusion matrix
  feature_importance.png    — Feature importance (RF)
  prediction_result.png     — Kết quả dự đoán
```

---

## Cấu Trúc Thư Mục

```
dsp-final/
├── app.py                     # Streamlit Web App (Equalizer + Classifier)
├── core/
│   ├── equalizer.py           # Logic FIR equalizer
│   └── ml_classifier.py       # Logic ML phân loại
├── models/                    # Model đã train (tạo từ notebook)
│   ├── genre_model.joblib
│   └── label_encoder.joblib
├── equalizer.ipynb            # Module 1: FIR Equalizer
├── ml_classification.ipynb    # Module 2: ML Genre Classifier
├── requirements.txt           # Danh sách thư viện
├── README.md                  # File này
├── SPEC.md                    # Kiến trúc & phân công
├── report-outline.md          # Outline báo cáo 15 trang
├── test-plan.md               # Kế hoạch test
├── train-data/
│   └── original-genres/
│       ├── blues/             # ~90 files .wav
│       ├── classical/
│       ├── country/
│       ├── disco/
│       ├── hiphop/
│       ├── jazz/
│       ├── metal/
│       ├── pop/
│       ├── reggae/
│       └── rock/
├── test-data/
│   ├── blues/                 # ~10 files .wav
│   └── ...
├── audio-equalizer-master/    # Code tham khảo (IIR)
└── music-equalizer/           # Code tham khảo (FFT)
```

---

## Thông Số Kỹ Thuật

### FIR Equalizer

| Thông số | Giá trị |
|----------|---------|
| Tần số lấy mẫu | 44,100 Hz |
| Phương pháp | FIR Bandpass, cửa sổ Hamming |
| Số taps | 1,025 |
| Số bands | 10 (20 Hz – 20 kHz) |
| Dải gain | -12 dB đến +12 dB |

### ML Classifier

| Thông số | Giá trị |
|----------|---------|
| Features | 82 (MFCC × 40, Chroma × 24, Contrast × 14, ZCR × 2, RMS × 2) |
| Models | SVM (RBF) + Random Forest (n=200) |
| Genres | 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) |

---

## Tài Liệu Tham Khảo

- SciPy `firwin`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
- Librosa: https://librosa.org/doc/latest/index.html
- GTZAN Dataset: Tzanetakis & Cook, 2002
