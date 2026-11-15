# Lịch sử cải tiến mô hình phát hiện gian lận thẻ tín dụng

- **Bài toán:** Kaggle Playground Series S6E11 — *Predicting Credit Card Fraud*
- **Metric đánh giá:** `ROC-AUC`
- **Tập dữ liệu:**
  - `data/train.csv`: `699.960` dòng, `24` đặc trưng gốc + `id` + `is_fraud` (~`1.67%` positive)
  - `data/test.csv`: `299.984` dòng, `24` đặc trưng gốc + `id`

---

## 1. Tóm tắt lộ trình điểm số (từ `LinearSVC` đến `0.95015`)

| Mốc | Commit | File chính | Kiến trúc & Đặc trưng chính | Điểm ghi nhận |
| :---: | :--- | :--- | :--- | :--- |
| 0 | `-` | `data/sample_submission.csv` | Dự đoán hằng số (`0.0166995...`) | `~0.50000` (Baseline ngẫu nhiên) |
| 1 | `3eb8938` | `code.py` | `LinearSVC` hiệu chỉnh xác suất (5-Fold, 38 cột tuyến tính) | `0.94294` (LB/CV) |
| 2 | `733212a` | `test.ipynb` | `XGBoost` + 24 cột gốc $\rightarrow$ 27 cột (Nhóm B Group Agg) | Fold 1: `0.94416 -> 0.94497` \| LB: `0.94695` |
| 3 | `1e1675e` | `test.ipynb` | Ensemble 2 mô hình (`XGB + LGB`, 27 cột) | OOF: `0.94674` \| LB: `0.94711` |
| 4 | `af2fb5e` | `notebooka55f0b78be.ipynb` | Ensemble 4 mô hình (`XGB + LGB + CAT + TabularResNet`, 31 cột) | OOF: `0.94686` \| LB: `0.94718` |
| 5 | `180a1cb` | `notebooka55f0b78be.ipynb` | Tinh chỉnh hyperparameters chống overfit (31 cột) | LB: `0.94728` (`+0.00010`) |
| 6 | *(local)* | `notebooka55f0b78be.ipynb` | **Shallow Trees + 36 Digit-Key Cross-fitted TE (60 cột) + Logit Blend** | Fold 1 Blend: `0.94850` \| **LB: `0.95015` (`+0.00287`)** |

---

## 2. Chi tiết từng giai đoạn & Bài học thực nghiệm

### Giai đoạn 1: `LinearSVC` Baseline (`code.py` | commit `3eb8938`) $\rightarrow$ `0.94294`
- **Pipeline:** `SimpleImputer` + `StandardScaler` + `OneHotEncoder` $\rightarrow$ `CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3, method="sigmoid")`, Stratified 5-Fold.
- **Đặc trưng (`38` cột):** `24` cột gốc + `14` cột biến đổi hỗ trợ mô hình tuyến tính (`amount_to_balance_ratio`, `txn_intensity`, `log_amount`, `log_distance`, `log_balance`, `risk_flag_count`, `has_cvv_retry`, `has_prior_dispute`, `amount_x_velocity`, `is_night`, `hour_sin`/`cos`, `dow_sin`/`cos`).
- **Ý nghĩa:** Ngay cả một mô hình tuyến tính đơn giản đã đạt `0.94294`, gợi ý quan trọng rằng tín hiệu chính của bộ dữ liệu S6E11 mang tính **cộng tính (additive)**.

---

### Giai đoạn 2: Chuyển sang GBDT & Feature Ablation (`test.ipynb` | `feature_ablation_report.txt`) $\rightarrow$ `0.94695` $\rightarrow$ `0.94711`
- **Thí nghiệm đối chứng trên Fold 1** (`XGBClassifier`, `max_depth=6`, `lr=0.05`):

| Thí nghiệm | Số cột | Fold 1 ROC-AUC | Nhận xét |
| :--- | :---: | :---: | :--- |
| `24` cột gốc thuần | `24` | `0.94416` | Mốc chuẩn GBDT |
| Bộ `38` cột thiết kế cho `LinearSVC` | `38` | `0.94420` | `+0.00004` (`log1p` không giúp cây) |
| Nhóm A (Ratios + Risk Sums) | `30` | `0.94424` | Gây nhiễu khi gộp cùng nhóm khác |
| Nhóm C (Frequency Encoding) | `29` | `0.94451` | Tăng khá khi đứng riêng |
| Nhóm D (Biến nhân rủi ro + `is_night`) | `35` | `0.94391` | `-0.00025` (làm giảm điểm) |
| Bộ `53` cột (`cat_channel` 60 loại rời rạc) | `53` | *(giảm)* | Phân mảnh category, giảm điểm Kaggle |
| **Nhóm B (3 cột Group Aggregations)** | **`27`** | **`0.94497`** | **`+0.00081` — tốt nhất, LB `0.94695`** |

- **3 cột Nhóm B:** `amount_ratio_cat`, `amount_ratio_ch`, `risk_diff_cat` (tính từ `X_tr` mỗi fold để tránh data leakage).
- **Ensemble 2 mô hình (`XGB + LGB`, 27 cột, commit `1e1675e`):**
  - Local OOF: `XGB = 0.94664`, `LGB = 0.94621`, `Ensemble = 0.94674` $\rightarrow$ **Kaggle LB `0.94711`**.

---

### Giai đoạn 3: Ensemble 4 kiến trúc khác nhau (`notebooka55f0b78be.ipynb` | commit `af2fb5e`) $\rightarrow$ `0.94718`
- Mở rộng Group Aggregations từ `3` lên `7` cột (`amount_ratio_cat`/`ch`/`auth`/`dev`, `risk_diff_cat`, `vel_diff_cat`/`ch` $\rightarrow$ `31` cột, tính theo từng fold).
- Bổ sung **CatBoost** (GPU) và **PyTorch `TabularResNet`** (`44` cột = `31` cột + `13` cột `add_nn_extra_features`, `QuantileTransformer` + `Categorical Embedding`).
- **Kết quả OOF trên Kaggle GPU:**
  - `XGB = 0.94669` | `LGB = 0.94640` | `CAT = 0.94491` | `NN = 0.94178`
  - `Overall OOF Ensemble = 0.94686` $\rightarrow$ **Kaggle LB `0.94718`**.

---

### Giai đoạn 4: Tinh chỉnh Hyperparameters chống Overfit (`notebooka55f0b78be.ipynb` | commit `180a1cb`) $\rightarrow$ `0.94728`
- Giữ nguyên `31` cột, siết regularization và độ phân giải bin:
  - `LR = 0.02` (GPU), `N_EST = 5000`, `EARLY_STOP = 120`.
  - `XGBoost`: `colsample_bytree` `0.75 -> 0.60`, `min_child_weight` `30 -> 20`, `max_depth=5`, `max_bin=512`.
  - `LightGBM`: `num_leaves` `31 -> 23`, `colsample_bytree` `0.75 -> 0.60`, `max_bin=511`.
  - `CatBoost`: `depth` `7 -> 6`, `l2_leaf_reg` `15 -> 10`, `eval_metric` `"Logloss" -> "AUC"`.
- **Kết quả:** Kaggle LB tăng từ `0.94718` $\rightarrow$ **`0.94728` (`+0.00010`)**.

---

### Giai đoạn 5: Đột phá — Shallow Trees + 36 Digit-Key Cross-fitted OOF TE $\rightarrow$ `0.95015` (`+0.00287`)

#### 5.1. Kiểm chứng loại bỏ 4 gói FE truyền thống trên Fold 1 (tránh tụt điểm)
Mốc so sánh trên Fold 1 (`31` cột): **`XGB = 0.94553` | `LGB = 0.94553`**.

| Thí nghiệm FE trên Fold 1 | Số cột | XGBoost | LightGBM | Kết luận |
| :--- | :---: | :---: | :---: | :--- |
| `+` Gói 1: OOF TE cặp category & `flag_pattern` | `37` | `0.94532` | `0.94501` | Giảm $\rightarrow$ Loại |
| `+` Gói 2: Group Z-score / Median / NewCols | `36` | `0.94518` | `0.94532` | Giảm $\rightarrow$ Loại |
| `+` Gói 3: Domain Consistency (tuổi thẻ, POS anomaly) | `36` | `0.94498` | `0.94526` | Giảm $\rightarrow$ Loại |
| `+` Gói 4: `cents` / `round` / `hour_cat` / `dow_cat` | `35` | `0.94451` | `0.94427` | Giảm mạnh $\rightarrow$ Loại |
| Gộp cả 4 gói | `51` | `0.94427` | `0.94393` | Giảm nặng $\rightarrow$ Loại |

#### 5.2. Phát hiện cấu trúc cộng tính $\rightarrow$ Thử nghiệm cây rất nông trên Fold 1
- Trên `31` cột:
  - `LightGBM`: `leaves=12` (`0.94540`) $\rightarrow$ `leaves=8` (`0.94577`) $\rightarrow$ `leaves=6` (`0.94603`) $\rightarrow$ **`leaves=4` (`0.94625`)** (`leaves=3`: `0.94611`; `leaves=2`: `0.94528` quá nông).
  - `XGBoost`: `depth=4` (`0.94564`) $\rightarrow$ `depth=3` (`0.94599`) $\rightarrow$ **`depth=2` (`0.94614`)**.
- Hạ độ sâu giúp tránh học tương tác nhiễu và tăng `+0.00061` đến `+0.00072` ngay trên Fold 1.

#### 5.3. Kết hợp `Digit Keys` + Cross-fitted OOF Target Encoding (`smooth=20.0`, lồng 5 fold trong `X_tr`)
- Tách `6` biến số thực thành **`36` khóa thành phần chữ số** (`_build_digit_keys`):
  - `4` biến `2` số lẻ (`amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`): `__int`, `__cents`, `__tenths`, `__hundredths`, `__ones`, `__tens`, `__lead` (`28` khóa).
  - `2` biến `1` số lẻ (`velocity_score`, `merchant_risk_score`): `__int`, `__tenths`, `__ones`, `__tens` (`8` khóa).
- Kết quả đối chứng trên Fold 1:
  - `31` cột `+ DigitTE` (`67` cột): `LGB leaves=6` (`0.94819`) $\rightarrow$ `leaves=4` (`0.94834`) $\rightarrow$ `leaves=3` (`0.94839`).
  - **`24` cột gốc `+ DigitTE` (`60` cột, bỏ 7 cột group cũ):**
    - `LGB leaves=2`: `0.94825` (`3137` vòng)
    - **`LGB leaves=3`: `0.94846` (`1880` vòng)** *(+`0.00293` so với `0.94553`)*
    - `XGB depth=3`: `0.94796` (`1030` vòng)
    - **`XGB depth=2`: `0.94835` (`1535` vòng)**
    - **Fold 1 Blend (`0.5 LGB + 0.5 XGB`):** `0.94850` (`prob`) / `0.94849` (`logit`).

#### 5.4. Thay đổi chốt trong `notebooka55f0b78be.ipynb` $\rightarrow$ Kaggle LB `0.95015`
- Thay `add_fold_group_features` (`31` cột) bằng `add_fold_digit_te` (`60` cột = `24` gốc + `36` Digit TE).
- Đặt cây rất nông: `XGB max_depth=2`, `LGB num_leaves=3`, `CatBoost depth=4`, `colsample_bytree=0.70`, `LR=0.025` (GPU), `N_EST=8000`, `EARLY_STOP=150`.
- Bổ sung không gian **`logit`** (`scipy.special.logit` / `expit`) cùng `prob` và `rank` (`400` ứng viên Dirichlet) ở bước tìm trọng số Ensemble.
- **Kết quả:** Kaggle LB tăng từ `0.94728` $\rightarrow$ **`0.95015` (`+0.00287`)**, khớp sát mức tăng đo trên Fold 1 (`+0.00297`).

---

## 3. Nguyên tắc cốt lõi rút ra từ quá trình tối ưu

1. **Kiểm tra Data Leakage trước tiên:** mọi thống kê nhóm và Target Encoding đều phải cross-fit lồng bên trong `X_tr` của từng fold.
2. **Không thêm FE theo cảm tính:** luôn chạy ablation đối chứng trên Fold 1 để loại bỏ kịp thời những gói gây giảm điểm (tránh rơi `-0.0012` AUC).
3. **Tôn trọng bản chất dữ liệu:** khi mục tiêu gần với hàm cộng tính có nhiễu, giảm độ sâu cây (`num_leaves=3`, `max_depth=2`) kết hợp Digit-Key TE mang lại bước nhảy vượt trội hơn hẳn việc trồng cây sâu hay ghép công thức thủ công.

