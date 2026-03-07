# Outline Báo Cáo Thuyết Minh
## Thiết Kế Bộ Cân Bằng Số (Equalizer) và Phân Loại Dòng Nhạc

**Độ dài:** 15 trang — Cỡ chữ: 12pt — Font: Times New Roman — Dãn dòng: 1.5

---

## TÓM TẮT (1/4 trang — ~½ cột A4)

**Bối cảnh:**
> Nhóm xây dựng ứng dụng xử lý tín hiệu âm thanh gồm 2 thành phần: (1) bộ cân bằng âm thanh số (graphic equalizer) 10 bands sử dụng bộ lọc FIR với cửa sổ Hamming, và (2) hệ thống phân loại thể loại âm nhạc sử dụng học máy (MFCC + SVM/Random Forest). Ứng dụng được triển khai trên Jupyter Notebook bằng Python.

**Kết quả đạt được:**
- ✅ Thiết kế thành công bộ FIR 10-band, áp dụng trên file .wav
- ✅ Trực quan hóa waveform và phổ tần số trước/sau equalizer
- ✅ Phân loại được 10 thể loại nhạc với độ chính xác ~X% (điền sau khi chạy)
- ✅ Xuất file âm thanh đã cân bằng

**Kết quả chưa đạt được:**
- ❌ Chưa xử lý realtime (chỉ dừng ở file-based processing)
- ❌ Chưa có GUI đồ họa độc lập (chạy trong Jupyter)

**Đóng góp thành viên:**

| Thành viên | Vai trò | Đóng góp |
|------------|---------|----------|
| [TV1] | DSP Lead | 25% — Thiết kế FIR filter, cấu hình bands |
| [TV2] | DSP / Visualization | 25% — Load audio, plot waveform/spectrum, output |
| [TV3] | ML Lead | 25% — Feature extraction, train model |
| [TV4] | ML / Báo cáo | 25% — Đánh giá model, viết báo cáo |

---

## 1. GIỚI THIỆU (2 trang)

### 1.1 Lựa Chọn và Định Hướng Thiết Kế

**Nội dung cần viết:**
- Lý do chọn đề tài (tầm quan trọng của xử lý âm thanh số)
- Nhóm lựa chọn bộ lọc **FIR** vì:
  - Ổn định vô điều kiện (không lo mất ổn định)
  - Có pha tuyến tính (không méo dạng sóng)
  - Dễ thiết kế và phân tích
- Nhóm chọn **Jupyter Notebook** vì:
  - Phù hợp cho trình bày học thuật
  - Tích hợp code + kết quả + giải thích
  - Dễ test và demo

### 1.2 Chỉ Tiêu Kỹ Thuật Hướng Đến

**Môi trường hoạt động:**
- Mở và phát nhạc từ file `.wav` trên máy tính
- Chạy trên Python 3.10+, môi trường Jupyter Notebook
- Không hỗ trợ realtime (file-based processing)

**Tính năng Equalizer:**
- 10 bands từ 20 Hz đến 20,000 Hz
- Điều chỉnh gain từ -12 dB đến +12 dB mỗi band
- Trực quan hóa waveform và phổ tần số
- Xuất file âm thanh đã xử lý

**Tính năng ML:**
- Phân loại 10 thể loại nhạc (GTZAN dataset)
- Hiển thị độ tin cậy (probability)
- So sánh 2 giải thuật: SVM và Random Forest

---

## 2. CƠ SỞ LÝ THUYẾT (3 trang)

### 2.1 Tín Hiệu Số và Lấy Mẫu

**Nội dung cần viết:**
- Khái niệm tín hiệu rời rạc, tần số lấy mẫu f_s
- Định lý Nyquist-Shannon: f_s ≥ 2 × f_max
- Với audio: f_s = 44,100 Hz (chuẩn CD) → biểu diễn tần số < 22,050 Hz
- Độ phân giải 16-bit: dải giá trị [-32768, 32767]

**Công thức:**
```
f_s = 44,100 Hz
T_s = 1/f_s ≈ 22.7 μs
Dải tần biểu diễn: 0 – 22,050 Hz
```

### 2.2 Bộ Lọc FIR

**Nội dung cần viết:**
- Định nghĩa FIR (Finite Impulse Response)
- Phương trình sai phân:
  ```
  y[n] = Σ(k=0 to M) b_k × x[n-k]
  ```
- Không có hồi tiếp → luôn ổn định
- Pha tuyến tính khi đối xứng: h[n] = h[M-n]
- Ưu/nhược điểm so với IIR

### 2.3 Phương Pháp Cửa Sổ (Windowed Sinc)

**Nội dung cần viết:**
- Bộ lọc lý tưởng → đáp ứng xung vô hạn (sinc) → cần cắt cụt
- Cửa sổ Hamming: giảm side-lobe tốt hơn rectangular
  ```
  w[n] = 0.54 - 0.46 × cos(2πn / M),   0 ≤ n ≤ M
  ```
- Hệ số FIR: h[n] = h_ideal[n] × w[n]
- Chọn **numtaps = 1025** (M=1024, lẻ → type I linear phase)

### 2.4 Phân Chia Băng Tần (10 Bands)

**Nội dung cần viết:**
- Bảng 10 bands (từ SPEC.md)
- Phương pháp: phân chia theo thang logarit (phù hợp nhận thức thính giác)
- Tần số cắt chuẩn hóa: f_norm = f / (f_s/2)

### 2.5 Cơ Sở Lý Thuyết Machine Learning

**Nội dung cần viết:**

**MFCC (Mel-Frequency Cepstral Coefficients):**
- Mô phỏng cách tai người nghe âm thanh (thang Mel phi tuyến)
- Quy trình: Audio → FFT → Mel filterbank → log → DCT → MFCC
- 20 hệ số MFCC, lấy mean và std → 40 features

**SVM (Support Vector Machine):**
- Tìm siêu phẳng phân tách tối ưu với margin lớn nhất
- Kernel RBF: K(x,x') = exp(-γ||x-x'||²)
- Phù hợp với feature vector nhỏ, nhiều classes

**Random Forest:**
- Ensemble của nhiều Decision Trees
- Voting đa số → giảm overfitting
- Feature importance → giải thích được

---

## 3. THIẾT KẾ (3 trang)

### 3.1 Thiết Kế Bộ Lọc FIR Cho Từng Band

**Nội dung cần viết + hình:**
- Vẽ frequency response của cả 10 FIR bandpass filters trên cùng 1 đồ thị
- Giải thích tại sao chọn numtaps = 1025
  - Nhiều tap hơn → dải chuyển tiếp hẹp hơn
  - Trade-off: độ trễ = numtaps/2 mẫu = 512/44100 ≈ 11.6 ms
- Code `scipy.signal.firwin()` với tham số

```python
# Ví dụ band 2 (60-170 Hz)
b = scipy.signal.firwin(1025, [60, 170], pass_zero=False,
                        window='hamming', fs=44100)
```

### 3.2 Sơ Đồ Tổ Chức Chương Trình

**Nội dung:**
- Vẽ flowchart hoặc mô tả dạng text (xem SPEC.md phần Data Flow)
- Gồm các bước: Load → Filter Bank → Gain Apply → Sum → Visualize → Save
- Module ML: Load dataset → Feature Extract → Train → Evaluate

### 3.3 Thiết Kế Pipeline ML

**Nội dung + hình:**
- Sơ đồ pipeline: Audio → Feature Extraction → StandardScaler → Model → Prediction
- Giải thích tại sao chuẩn hóa (StandardScaler) là cần thiết
- Giải thích lý do chọn train/test split (hoặc dùng dataset có sẵn)

### 3.4 Phương Pháp Test

**Test module equalizer:**
- Test với tín hiệu sine đơn tần (kiểm tra 1 band hoạt động đúng)
- Test với white noise (kiểm tra tất cả bands)
- Test với file nhạc thực

**Test module ML:**
- Dùng test-data/ (tập test riêng biệt)
- Đánh giá: accuracy, precision, recall, F1
- Confusion matrix để xem lỗi phân loại

---

## 4. KẾT QUẢ THỰC HIỆN (4 trang)

### 4.1 Kết Quả Module Equalizer

**Hình ảnh cần có (chụp screenshot từ notebook):**

1. **Frequency Response của 10 bands** (1 đồ thị, 10 đường)
   - Trục x: tần số (Hz, log scale)
   - Trục y: Gain (dB)
   - Giải thích: mỗi band có dải thông rõ ràng, không chồng lấn nhiều

2. **Waveform trước và sau equalizer** (2 đồ thị chồng hoặc subplot)
   - Trục x: thời gian (giây)
   - Trục y: biên độ
   - Giải thích: sự thay đổi biên độ khi tăng/giảm bands

3. **Phổ tần số FFT trước và sau** (2 đường trên cùng đồ thị)
   - Trục x: tần số (Hz, log scale)
   - Trục y: Magnitude (dB)
   - Giải thích: thể hiện rõ bands nào được tăng/giảm

**Code cần show (trích đoạn quan trọng):**
```python
# Thiết kế filter
def design_fir_bandpass(low_hz, high_hz, fs, numtaps=1025):
    b = signal.firwin(numtaps, [low_hz, high_hz],
                      pass_zero=False, window='hamming', fs=fs)
    return b

# Áp dụng equalizer
def apply_equalizer(audio, fs, gains_db, bands, numtaps=1025):
    output = np.zeros_like(audio)
    for (low, high), gain_db in zip(bands, gains_db):
        b = design_fir_bandpass(low, high, fs, numtaps)
        filtered = signal.lfilter(b, [1.0], audio)
        gain_linear = 10 ** (gain_db / 20)
        output += filtered * gain_linear
    return output
```

### 4.2 Kết Quả Module ML

**Hình ảnh cần có:**

1. **Confusion Matrix** của SVM và Random Forest (2 heatmaps)
   - Giải thích: genre nào bị nhầm lẫn nhiều nhất và tại sao

2. **Bar chart so sánh accuracy** SVM vs Random Forest

3. **Classification Report** (in ra dạng bảng)

**Biện luận kết quả:**
- Accuracy đạt được: X% (SVM), Y% (RF)
- Nhạc thể loại nào khó phân loại nhất? Tại sao?
- (Ví dụ: country vs rock có thể nhầm vì nhịp điệu tương tự)

### 4.3 Biện Luận

**Kết quả đạt được:**
- Bộ lọc FIR hoạt động đúng (frequency response phù hợp thiết kế)
- ...

**Kết quả chưa đạt được / Hạn chế:**
- Chưa xử lý realtime (yêu cầu thêm PyAudio + threading)
- Chưa có GUI tương tác (chỉ Jupyter)
- ...

---

## 5. PHẢN ÁNH KẾT QUẢ CÁ NHÂN (1 trang — mỗi người ~¼ trang)

> Mỗi thành viên viết phần của mình.

**Thành viên 1 ([Tên]) — DSP Lead:**
- Công việc được giao: ...
- Kết quả thực hiện: ...
- Khó khăn gặp phải: ...
- Bài học rút ra: ...

**Thành viên 2 ([Tên]) — DSP / Visualization:**
- ...

**Thành viên 3 ([Tên]) — ML Lead:**
- ...

**Thành viên 4 ([Tên]) — ML / Báo cáo:**
- ...

---

## 6. KẾT LUẬN (½ trang)

**Nội dung:**
- Tóm tắt lại những gì đã làm được
- Nhận xét tổng quan về phương pháp FIR cho equalizer
- Nhận xét về hiệu quả của ML cho phân loại nhạc
- Hướng phát triển tiếp theo (nếu có thêm thời gian):
  - Thêm realtime processing với PyAudio
  - Thử deep learning (CNN trên Mel spectrogram)
  - Đóng gói thành web app

---

## TÀI LIỆU THAM KHẢO

1. Proakis, J.G., Manolakis, D.G. *Digital Signal Processing*, 4th Ed. Pearson, 2007.
2. Oppenheim, A.V., Schafer, R.W. *Discrete-Time Signal Processing*, 3rd Ed. Pearson, 2010.
3. SciPy Documentation — `scipy.signal.firwin`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
4. Librosa Documentation: https://librosa.org/doc/latest/index.html
5. Tzanetakis, G., Cook, P. "Musical genre classification of audio signals." *IEEE Trans. Speech Audio Process.*, 2002.
6. Tài liệu khóa học DSP — Lecture 4: *Lộ trình khái niệm DSP cho bài lab lọc FIR*

---

## GHI CHÚ KHI VIẾT

- **Hình ảnh:** đánh số (Hình 1, Hình 2...) và có chú thích bên dưới
- **Công thức:** đánh số (1), (2)... căn lề phải
- **Code trích dẫn:** dùng font Courier New, cỡ 10pt, highlight keyword
- **Bảng:** đánh số (Bảng 1...) và có tiêu đề bên trên
- **Trang bìa** không tính vào 15 trang
- **Phụ lục** (nếu cần): để toàn bộ code đầy đủ ở phụ lục, không tính vào 15 trang

### Phân Bổ Trang

| Mục | Trang |
|-----|-------|
| Tóm tắt | 0.25 |
| 1. Giới thiệu | 2 |
| 2. Cơ sở lý thuyết | 3 |
| 3. Thiết kế | 3 |
| 4. Kết quả | 4 |
| 5. Phản ánh cá nhân | 1 |
| 6. Kết luận | 0.5 |
| Tài liệu tham khảo | 0.25 |
| **Tổng** | **~14 trang** |

> Thêm hình ảnh để đủ 15 trang nếu cần.
