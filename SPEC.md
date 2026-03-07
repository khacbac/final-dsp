# Project Specification
## Thiết kế Bộ Cân Bằng Số (Equalizer) và Phân Loại Dòng Nhạc

**Môn:** Xử Lý Tín Hiệu Số (DSP)
**Nhóm:** 4 thành viên
**Ngôn ngữ:** Python (DSP + ML), Jupyter Notebook (GUI)

---

## 1. Tổng Quan Hệ Thống

Hệ thống gồm 2 module độc lập, chạy trong Jupyter Notebook:

| Module | Mô tả | File |
|--------|-------|------|
| **Equalizer** | FIR 10-band graphic equalizer cho file audio | `equalizer.ipynb` |
| **ML Classifier** | Phân loại thể loại nhạc (10 genres) | `ml_classification.ipynb` |

---

## 2. Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────┐
│                    equalizer.ipynb                      │
│                                                         │
│  [Audio File (.wav/.mp3)]                               │
│         │                                               │
│         ▼                                               │
│  [Load & Preprocess]  ──► fs=44100Hz, int16→float32    │
│         │                                               │
│         ▼                                               │
│  [FIR Filter Bank]  ◄── 10 bands, Hamming window       │
│    Band 1: 20-60 Hz    (Sub-bass)                       │
│    Band 2: 60-170 Hz   (Bass)                           │
│    Band 3: 170-310 Hz  (Low-mid)                        │
│    Band 4: 310-600 Hz  (Mid)                            │
│    Band 5: 600-1k Hz   (Upper-mid)                      │
│    Band 6: 1k-3k Hz    (Presence)                       │
│    Band 7: 3k-6k Hz    (Brilliance)                     │
│    Band 8: 6k-10k Hz   (Air)                            │
│    Band 9: 10k-16k Hz  (High air)                       │
│    Band 10: 16k-20k Hz (Ultra-high)                     │
│         │                                               │
│         ▼                                               │
│  [Gain Apply]  ◄── slider: -12dB to +12dB per band     │
│         │                                               │
│         ▼                                               │
│  [Output Signal]                                        │
│    ├── Waveform plot (before vs after)                  │
│    ├── Spectrum plot (before vs after)                  │
│    └── Save to .wav file                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 ml_classification.ipynb                  │
│                                                         │
│  [Dataset: train-data/original-genres/]                 │
│    10 genres × 90 files = 900 training samples          │
│         │                                               │
│         ▼                                               │
│  [Feature Extraction]  (librosa)                        │
│    ├── MFCC (20 coefficients, mean+std = 40 features)  │
│    ├── Chroma STFT (12, mean+std = 24 features)         │
│    ├── Spectral Contrast (7, mean+std = 14 features)    │
│    ├── ZCR (mean+std = 2 features)                      │
│    └── RMS Energy (mean+std = 2 features)               │
│    Total: 82 features per sample                        │
│         │                                               │
│         ▼                                               │
│  [Model Training]                                       │
│    ├── SVM (RBF kernel, C=10, gamma=auto)               │
│    └── Random Forest (n=200, max_depth=None)            │
│         │                                               │
│         ▼                                               │
│  [Evaluation]  (test-data/)                             │
│    ├── Accuracy score                                   │
│    ├── Confusion matrix                                 │
│    └── Classification report                            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Chi Tiết Module Equalizer

### 3.1 Thông Số Kỹ Thuật

| Thông số | Giá trị |
|----------|---------|
| Tần số lấy mẫu | 44,100 Hz (CD quality) |
| Độ phân giải | 16-bit signed int |
| Số kênh | Stereo (xử lý từng kênh) hoặc Mono |
| Số bands | 10 |
| Loại bộ lọc | FIR Bandpass |
| Cửa sổ | Hamming |
| Số taps | 1025 (N phải lẻ để linear phase) |
| Dải gain | -12 dB đến +12 dB |

### 3.2 Phân Chia Băng Tần

| Band | Tên | Dải tần | Mô tả |
|------|-----|---------|-------|
| 1 | Sub-bass | 20 – 60 Hz | Tiếng trầm sâu, kick drum |
| 2 | Bass | 60 – 170 Hz | Tiếng bass guitar, warmth |
| 3 | Low-mid | 170 – 310 Hz | Thân âm nhạc cụ |
| 4 | Mid | 310 – 600 Hz | Vùng giữa, giọng nói |
| 5 | Upper-mid | 600 – 1,000 Hz | Độ rõ ràng giọng hát |
| 6 | Presence | 1,000 – 3,000 Hz | Độ hiện diện, chi tiết |
| 7 | Brilliance | 3,000 – 6,000 Hz | Độ sắc nét |
| 8 | Air | 6,000 – 10,000 Hz | Độ thoáng, sáng |
| 9 | High air | 10,000 – 16,000 Hz | Tần số rất cao |
| 10 | Ultra-high | 16,000 – 20,000 Hz | Giới hạn nghe của người |

### 3.3 Công Thức FIR

Thiết kế bằng phương pháp cửa sổ (Windowed Sinc):

```
y[n] = Σ(k=0 to M) b_k × x[n-k]
```

Trong đó:
- `b_k`: hệ số FIR (tap coefficients)
- `M = numtaps - 1 = 1024`: bậc bộ lọc
- Tần số cắt chuẩn hóa: `ω_c = 2π × f_c / f_s`

Áp dụng gain theo dB:
```
gain_linear = 10^(gain_dB / 20)
y_band[n] = gain_linear × FIR_bandpass(x[n])
output[n] = Σ(i=1 to 10) y_band_i[n]
```

---

## 4. Chi Tiết Module ML

### 4.1 Dataset

- **Train:** `train-data/original-genres/` — 10 genres × ~90 files = ~900 samples
- **Test:** `test-data/` — 10 genres × ~10 files = ~100 samples
- **Genres:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **Format:** `.wav`, 30 giây/file, 22050 Hz

### 4.2 Feature Vector (82 features)

| Feature | Dim | Cách tính |
|---------|-----|-----------|
| MFCC | 40 | 20 coef × (mean + std) |
| Chroma STFT | 24 | 12 × (mean + std) |
| Spectral Contrast | 14 | 7 × (mean + std) |
| ZCR | 2 | mean + std |
| RMS Energy | 2 | mean + std |

### 4.3 Thuật Toán

**SVM (Support Vector Machine):**
- Kernel: RBF (Radial Basis Function)
- C = 10, gamma = 'scale'
- Preprocessing: StandardScaler

**Random Forest:**
- n_estimators = 200
- max_depth = None (unlimited)
- Preprocessing: StandardScaler

---

## 5. Phân Công Công Việc

| Thành viên | Role | Nhiệm vụ |
|------------|------|----------|
| **TV1** | DSP Lead | FIR filter design, band configuration, filter testing |
| **TV2** | DSP / Viz | Audio loading, waveform/spectrum visualization, output |
| **TV3** | ML Lead | Feature extraction (MFCC, Chroma...), model training |
| **TV4** | ML / Report | Model evaluation, confusion matrix, viết báo cáo tổng hợp |

> **Lưu ý:** TV1 và TV2 làm `equalizer.ipynb`, TV3 và TV4 làm `ml_classification.ipynb`. Mỗi người viết phần "Phản ánh kết quả cá nhân" trong báo cáo.

---

## 6. Tech Stack

```
Python 3.10+
├── numpy          # Xử lý mảng số
├── scipy          # Thiết kế bộ lọc FIR (firwin, sosfilt)
├── librosa        # Load audio, MFCC, Chroma, Spectral features
├── soundfile      # Đọc/ghi WAV file
├── matplotlib     # Vẽ waveform, spectrum, confusion matrix
├── scikit-learn   # SVM, Random Forest, StandardScaler, metrics
├── ipywidgets     # Interactive sliders trong Jupyter (điểm cộng)
└── jupyter        # Môi trường notebook
```

Cài đặt:
```bash
pip install numpy scipy librosa soundfile matplotlib scikit-learn ipywidgets
```

---

## 7. Cấu Trúc Thư Mục

```
dsp-final/
├── equalizer.ipynb          # Module 1: FIR Equalizer
├── ml_classification.ipynb  # Module 2: ML Genre Classifier
├── SPEC.md                  # Tài liệu này
├── report-outline.md        # Outline báo cáo
├── test-plan.md             # Kế hoạch test
├── train-data/              # Dataset training (10 genres)
│   └── original-genres/
│       ├── blues/
│       ├── classical/
│       └── ...
├── test-data/               # Dataset testing
│   ├── blues/
│   ├── classical/
│   └── ...
├── audio-equalizer-master/  # Code tham khảo (IIR)
└── music-equalizer/         # Code tham khảo (FFT)
```

---

## 8. Luồng Dữ Liệu (Data Flow)

```
Audio File (.wav)
    │
    ▼  soundfile.read()
[numpy array: float32, shape=(N,) hoặc (N,2)]
    │
    ▼  nếu stereo → lấy kênh trái ([:,0]) hoặc mono
[mono signal: float32, shape=(N,)]
    │
    ├──────────────────────────────────┐
    │ (original)                       │ (cho từng band i)
    ▼                                  ▼
[Waveform Plot]          scipy.signal.firwin(numtaps, [low,high],
                              pass_zero=False, window='hamming', fs=fs)
                                       │
                                       ▼
                         scipy.signal.lfilter(b, [1.0], signal)
                                       │
                                       ▼
                         band_output × gain_linear[i]
                                       │
    ┌──────────────────────────────────┘
    ▼
[Sum all bands → output signal]
    │
    ├── Waveform Plot (before & after, overlay)
    ├── FFT Spectrum Plot (before & after)
    └── soundfile.write('output_eq.wav', output, fs)
```

---

## 9. Checklist Hoàn Thành

### Module Equalizer (bắt buộc)
- [ ] Load file .wav thành công
- [ ] Thiết kế được 10 FIR bandpass filters
- [ ] Plot frequency response của từng band
- [ ] Áp dụng gain và tính tín hiệu đầu ra
- [ ] Plot waveform trước và sau equalizer
- [ ] Plot phổ tần số (FFT) trước và sau
- [ ] Lưu file .wav đầu ra

### Module Equalizer (điểm cộng)
- [ ] Interactive sliders (ipywidgets)
- [ ] Real-time preview

### Module ML (bắt buộc)
- [ ] Load dataset và extract features
- [ ] Train SVM model
- [ ] Train Random Forest model
- [ ] Accuracy trên test set ≥ 70%
- [ ] Confusion matrix visualization
- [ ] Predict thể loại của 1 file audio mới

### Báo cáo (bắt buộc)
- [ ] Đủ 15 trang, cỡ chữ 12pt
- [ ] Có tóm tắt, giới thiệu, cơ sở lý thuyết
- [ ] Có hình ảnh kết quả test
- [ ] Có phần cá nhân của từng thành viên
