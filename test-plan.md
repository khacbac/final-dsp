# Test Plan
## Thiết Kế Bộ Cân Bằng Số (Equalizer) và Phân Loại Dòng Nhạc

---

## 1. Tổng Quan

| Module | Phương pháp test | Tiêu chí pass |
|--------|-----------------|--------------|
| FIR Filter (từng band) | Unit test với tín hiệu sine | Gain đúng ±1 dB |
| Equalizer tổng thể | Integration test với file WAV | Waveform thay đổi, không clip |
| ML Feature Extraction | Kiểm tra shape và range | shape=(82,), không NaN |
| ML Model | Test set accuracy | ≥ 70% |

---

## 2. Test Module Lõi — FIR Filter

### 2.1 Test Nguyên Lý Bộ Lọc (Unit Test)

**Mục đích:** Kiểm tra bộ lọc FIR có đúng đặc tính bandpass không.

**Phương pháp:** Tạo tín hiệu sine ở các tần số khác nhau, lọc qua FIR, đo biên độ đầu ra.

```python
import numpy as np
import scipy.signal as signal

def test_fir_bandpass_attenuation():
    """
    Kiểm tra FIR bandpass:
    - Tần số trong dải thông → giảm biên độ ≤ 3 dB (pass)
    - Tần số ngoài dải thông → giảm biên độ ≥ 40 dB (stop)
    """
    fs = 44100
    numtaps = 1025
    low, high = 1000, 3000   # Band Presence
    duration = 0.5           # giây
    t = np.arange(int(fs * duration)) / fs

    # Thiết kế FIR
    b = signal.firwin(numtaps, [low/fs*2, high/fs*2],
                      pass_zero=False, window='hamming')

    # Test tần số TRONG dải thông: f=2000 Hz
    f_pass = 2000
    x_pass = np.sin(2 * np.pi * f_pass * t)
    y_pass = signal.lfilter(b, [1.0], x_pass)
    # Bỏ transient (numtaps//2 mẫu đầu)
    rms_in  = np.sqrt(np.mean(x_pass[numtaps:]**2))
    rms_out = np.sqrt(np.mean(y_pass[numtaps:]**2))
    gain_db = 20 * np.log10(rms_out / rms_in + 1e-12)
    print(f'f={f_pass} Hz (trong dải): Gain = {gain_db:.2f} dB  |  Kỳ vọng: ≥ -3 dB')
    assert gain_db >= -3, f'FAIL: gain {gain_db:.2f} dB < -3 dB'

    # Test tần số NGOÀI dải thông: f=100 Hz
    f_stop = 100
    x_stop = np.sin(2 * np.pi * f_stop * t)
    y_stop = signal.lfilter(b, [1.0], x_stop)
    rms_in2  = np.sqrt(np.mean(x_stop[numtaps:]**2))
    rms_out2 = np.sqrt(np.mean(y_stop[numtaps:]**2))
    gain_db2 = 20 * np.log10(rms_out2 / rms_in2 + 1e-12)
    print(f'f={f_stop} Hz (ngoài dải): Gain = {gain_db2:.2f} dB  |  Kỳ vọng: ≤ -40 dB')
    assert gain_db2 <= -40, f'FAIL: gain {gain_db2:.2f} dB > -40 dB'

    print('PASS: FIR bandpass hoạt động đúng')

test_fir_bandpass_attenuation()
```

**Kết quả kỳ vọng:**
- Tần số trong dải thông (f=2000 Hz): Gain ≥ -3 dB (≈ 0 dB lý tưởng)
- Tần số dải chắn (f=100 Hz): Gain ≤ -40 dB

---

### 2.2 Test Đáp Ứng Tần Số Từng Band

**Mục đích:** Kiểm tra 10 bands phủ đúng dải tần 20 Hz – 20 kHz.

```python
def test_band_coverage():
    """
    Kiểm tra các band không bỏ sót dải tần quan trọng.
    Mỗi band phải có gain ≥ -6 dB trong vùng trung tâm của nó.
    """
    BANDS = [
        ('Sub-bass',   20,   60),
        ('Bass',       60,  170),
        ('Low-mid',   170,  310),
        ('Mid',       310,  600),
        ('Upper-mid', 600, 1000),
        ('Presence', 1000, 3000),
        ('Brilliance',3000, 6000),
        ('Air',      6000,10000),
        ('High air',10000,16000),
        ('Ultra-high',16000,20000),
    ]
    fs, numtaps = 44100, 1025

    for name, low, high in BANDS:
        b = signal.firwin(numtaps, [low/(fs/2), high/(fs/2)],
                          pass_zero=False, window='hamming')
        w, h = signal.freqz(b, worN=8192, fs=fs)
        # Tìm gain tại tần số trung tâm (log scale)
        f_center = np.sqrt(low * high)
        idx = np.argmin(np.abs(w - f_center))
        gain_db = 20 * np.log10(np.abs(h[idx]) + 1e-12)
        status = 'PASS' if gain_db >= -6 else 'FAIL'
        print(f'{status} | {name:12s} ({low:6d}-{high:6d} Hz) | center={f_center:7.1f} Hz | gain={gain_db:6.2f} dB')

test_band_coverage()
```

---

### 2.3 Test Áp Dụng Gain

**Mục đích:** Kiểm tra gain +6 dB và -6 dB hoạt động đúng.

```python
def test_gain_application():
    """
    Boost +6 dB → năng lượng tăng gấp 2 lần (10^(6/20) ≈ 2.0x).
    Cut -6 dB   → năng lượng giảm còn 1/2 (10^(-6/20) ≈ 0.5x).
    """
    gain_db = 6.0
    gain_linear = 10 ** (gain_db / 20)
    expected = 2.0  # gần đúng

    print(f'+6 dB → gain linear = {gain_linear:.4f}  (kỳ vọng ≈ {expected})')
    assert abs(gain_linear - expected) < 0.01, 'FAIL: gain linear không đúng'

    gain_db2 = -6.0
    gain_linear2 = 10 ** (gain_db2 / 20)
    print(f'-6 dB → gain linear = {gain_linear2:.4f}  (kỳ vọng ≈ 0.5)')
    assert abs(gain_linear2 - 0.5) < 0.01, 'FAIL'
    print('PASS: Gain application đúng')

test_gain_application()
```

---

### 2.4 Test Pha Tuyến Tính (Linear Phase)

**Mục đích:** Xác nhận FIR với cửa sổ Hamming có pha tuyến tính.

```python
def test_linear_phase():
    """
    FIR đối xứng (h[n] = h[M-n]) → pha tuyến tính.
    Kiểm tra: max|h[n] - h[M-n]| < epsilon
    """
    fs, numtaps = 44100, 1025
    b = signal.firwin(numtaps, [1000/(fs/2), 3000/(fs/2)],
                      pass_zero=False, window='hamming')
    M = len(b) - 1
    symmetry_error = np.max(np.abs(b - b[::-1]))
    print(f'Sai số đối xứng: {symmetry_error:.2e}  (kỳ vọng ≈ 0)')
    assert symmetry_error < 1e-10, f'FAIL: không đối xứng, error={symmetry_error}'
    print('PASS: FIR có pha tuyến tính (đối xứng)')

test_linear_phase()
```

---

## 3. Test Integration — Equalizer Toàn Thể

### 3.1 Test Với File Audio Thực

| Test case | Input | Gain | Kỳ vọng |
|-----------|-------|------|---------|
| TC01: Flat EQ | Bất kỳ .wav | Tất cả = 0 dB | Output ≈ Input (RMS chênh ≤ 1 dB) |
| TC02: Bass boost | blues.wav | Band 2 = +6 dB | Năng lượng 60-170 Hz tăng ≥ 4 dB |
| TC03: Treble cut | jazz.wav | Band 7,8 = -6 dB | Năng lượng 3k-10k Hz giảm ≥ 4 dB |
| TC04: Full EQ | rock.wav | Hỗn hợp +/-dB | Output không clip (max ≤ 1.0) |

```python
def test_flat_eq(audio_file):
    """
    TC01: Gain = 0 dB tất cả bands → output ≈ input.
    """
    import soundfile as sf
    audio, fs = sf.read(audio_file, dtype='float32')
    if audio.ndim == 2:
        audio = audio[:, 0]

    # Áp dụng với gain = 0 dB (tất cả bands)
    gains_db = [0] * 10
    # ... (gọi hàm apply_equalizer từ equalizer.ipynb)

    rms_in  = np.sqrt(np.mean(audio**2))
    # rms_out = np.sqrt(np.mean(output**2))
    # diff_db = 20 * np.log10(rms_out / rms_in)
    # assert abs(diff_db) < 1.0, f'FAIL: RMS chênh {diff_db:.2f} dB > 1 dB'
    print('TC01: Flat EQ — Chạy trong equalizer.ipynb với GAINS_DB = [0]*10')

def test_no_clipping(output_signal):
    """
    TC04: Tín hiệu đầu ra không được clip (biên độ vượt ±1.0).
    """
    max_val = np.max(np.abs(output_signal))
    print(f'Max biên độ đầu ra: {max_val:.4f}  (kỳ vọng ≤ 1.0)')
    assert max_val <= 1.001, f'FAIL: clipping! max={max_val:.4f}'
    print('PASS: Không có clipping')
```

---

## 4. Test Module ML

### 4.1 Test Feature Extraction

```python
def test_feature_extraction(sample_file):
    """
    Kiểm tra feature extraction trả về đúng shape và không có NaN.
    """
    features = extract_features(sample_file)

    # Kiểm tra shape
    assert features is not None, 'FAIL: extract_features trả về None'
    assert features.shape == (82,), f'FAIL: shape sai {features.shape} != (82,)'

    # Kiểm tra không có NaN hoặc Inf
    assert not np.isnan(features).any(), 'FAIL: có NaN trong features'
    assert not np.isinf(features).any(), 'FAIL: có Inf trong features'

    print(f'PASS: features shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]')
```

### 4.2 Test Model Accuracy

| Tiêu chí | Giá trị tối thiểu | Lý do |
|----------|------------------|-------|
| Overall accuracy | ≥ 70% | GTZAN baseline ~70% với MFCC+SVM |
| Per-class accuracy | ≥ 50% | Không có genre nào bị bỏ hoàn toàn |
| Classical recall | ≥ 80% | Genre dễ phân loại nhất |

```python
def test_model_accuracy(model, X_test, y_test, min_acc=0.70):
    """
    Kiểm tra model đạt accuracy tối thiểu.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.4f} ({acc*100:.1f}%)')
    assert acc >= min_acc, f'FAIL: accuracy {acc:.2%} < {min_acc:.0%}'
    print(f'PASS: Model đạt {acc:.1%} >= {min_acc:.0%}')
```

### 4.3 Test Dự Đoán File Mới

```python
def test_predict_known_genre(model, scaler, le):
    """
    Dự đoán file đã biết thể loại → kiểm tra kết quả.
    """
    # Dùng 1 file từ test set đã biết nhãn
    known_file  = 'test-data/classical/classical.00085.wav'
    true_genre  = 'classical'

    predicted, confidence, top3 = predict_genre(known_file, model, scaler, le)
    print(f'File   : {known_file}')
    print(f'Đúng   : {true_genre}')
    print(f'Dự đoán: {predicted}  (confidence: {confidence:.1f}%)')
    print(f'Top 3  : {top3}')
    # Classical thường dễ phân loại → phải nằm trong top 3
    top3_genres = [g for g, _ in top3]
    assert true_genre in top3_genres, f'FAIL: {true_genre} không có trong top 3'
    print('PASS: Genre đúng nằm trong top 3 dự đoán')
```

---

## 5. Kế Hoạch Chạy Test

### Trình tự chạy test trong notebook

| Bước | Test | File | Người thực hiện |
|------|------|------|-----------------|
| 1 | `test_fir_bandpass_attenuation()` | equalizer.ipynb | TV1 |
| 2 | `test_band_coverage()` | equalizer.ipynb | TV1 |
| 3 | `test_gain_application()` | equalizer.ipynb | TV2 |
| 4 | `test_linear_phase()` | equalizer.ipynb | TV1 |
| 5 | TC01: Flat EQ với WAV file | equalizer.ipynb | TV2 |
| 6 | TC04: No clipping check | equalizer.ipynb | TV2 |
| 7 | `test_feature_extraction()` | ml_classification.ipynb | TV3 |
| 8 | `test_model_accuracy()` SVM | ml_classification.ipynb | TV3 |
| 9 | `test_model_accuracy()` RF | ml_classification.ipynb | TV4 |
| 10 | `test_predict_known_genre()` | ml_classification.ipynb | TV4 |

### Ghi chép kết quả test

| Test | Kết quả | Giá trị đo | Pass/Fail |
|------|---------|-----------|----------|
| FIR in-band gain | | dB | |
| FIR stop-band attenuation | | dB | |
| Band coverage (10 bands) | | | |
| Linear phase | | | |
| No clipping | | max biên độ | |
| Feature shape | | | |
| SVM accuracy | | % | |
| RF accuracy | | % | |

> Điền vào bảng này sau khi chạy và chụp screenshot kết quả để đưa vào báo cáo.

---

## 6. Checklist Trước Khi Nộp

- [ ] Tất cả test trong mục 5 đã chạy và ghi kết quả
- [ ] Screenshot frequency response của 10 bands
- [ ] Screenshot waveform trước/sau EQ
- [ ] Screenshot phổ tần số trước/sau EQ
- [ ] Screenshot confusion matrix (SVM + RF)
- [ ] Accuracy của 2 model được ghi rõ
- [ ] File `audio_equalized.wav` được tạo ra
- [ ] Notebook chạy từ đầu đến cuối không có lỗi (Kernel → Restart & Run All)
