# DSP Final Project Memory

## Project Info

- Môn: Xử Lý Tín Hiệu Số (DSP)
- Chủ đề: FIR Graphic Equalizer + ML Genre Classification
- Nhóm: 4 thành viên (2 DSP, 2 ML)
- GUI: Jupyter Notebook

## Files Đã Tạo

| File                      | Mục đích                                           |
| ------------------------- | -------------------------------------------------- |
| `SPEC.md`                 | Kiến trúc, phân công, data flow, thông số kỹ thuật |
| `report-outline.md`       | Outline báo cáo 15 trang chi tiết                  |
| `equalizer.ipynb`         | FIR 10-band equalizer hoàn chỉnh                   |
| `ml_classification.ipynb` | MFCC + SVM/RF genre classification                 |
| `test-plan.md`            | Unit test + integration test plan                  |

## Key Technical Decisions

- **Filter:** FIR Bandpass, Hamming window, numtaps=1025
- **Bands:** 10 bands từ 20Hz–20kHz (log scale)
- **ML features:** MFCC(40) + Chroma(24) + SpectralContrast(14) + ZCR(2) + RMS(2) = 82 features
- **ML models:** SVM (RBF, C=10) và Random Forest (n=200)
- **Dataset:** train-data/genres_original/ (10 genres, ~900 files), test-data/ (~100 files)

## Code Reference

- `audio-equalizer-master/equalizer.py` — IIR (Butterworth) bandpass, 6 bands, Tkinter GUI
- `music-equalizer/processfunc.py` — FFT-based equalizer
