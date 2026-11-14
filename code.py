import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển kiểu boolean và tạo thêm các cột đặc trưng hỗ trợ mô hình tuyến tính."""
    out = df.copy()
    out[BOOL_COLS] = out[BOOL_COLS].astype(int)

    # Tỷ lệ số tiền giao dịch so với số dư & mật độ giao dịch
    out["amount_to_balance_ratio"] = out["amount_usd"] / (out["account_balance_usd"].clip(lower=0) + 1.0)
    out["txn_intensity"] = out["txn_count_last_24h"] / (out["hours_since_last_txn"].clip(lower=0) + 0.1)

    # Biến đổi log giảm lệch phân phối cho SVM
    out["log_amount"] = np.log1p(out["amount_usd"].clip(lower=0))
    out["log_distance"] = np.log1p(out["distance_from_home_km"].clip(lower=0))
    out["log_balance"] = np.log1p(out["account_balance_usd"].clip(lower=0))

    # Tổng hợp dấu hiệu rủi ro & tương tác
    out["risk_flag_count"] = out[BOOL_COLS].sum(axis=1)
    out["has_cvv_retry"] = (out["cvv_retry_count"] > 0).astype(int)
    out["has_prior_dispute"] = (out["prior_disputes"] > 0).astype(int)
    out["amount_x_velocity"] = out["log_amount"] * out["velocity_score"]

    # Mã hóa chu kỳ thời gian (giúp SVM hiểu 23h gần 0h, Chủ nhật gần Thứ 2)
    out["is_night"] = ((out["time_of_day_hour"] < 6) | (out["time_of_day_hour"] >= 23)).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["time_of_day_hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["time_of_day_hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    return out


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Tạo pipeline tiền xử lý + LinearSVC hiệu chỉnh xác suất."""
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    svm_model = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42),
        cv=3,
        method="sigmoid",
    )

    return Pipeline([
        ("prep", preprocessor),
        ("model", svm_model),
    ])


# 1. Đọc dữ liệu & tạo thêm các cột mới
train = add_features(pd.read_csv("data/train.csv"))
test = add_features(pd.read_csv("data/test.csv"))

X = train.drop(columns=["id", "is_fraud"])
y = train["is_fraud"].astype(int)
X_test = test.drop(columns=["id"])

num_cols = [c for c in X.columns if c not in CAT_COLS]
print(f"Số đặc trưng sau khi bổ sung: {X.shape[1]} cột ({len(num_cols)} số/bool + {len(CAT_COLS)} phân loại)")

# 2. Cross-Validation (Stratified 5-Fold)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros(len(X))
test_probs = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    pipe = build_pipeline(num_cols, CAT_COLS)
    pipe.fit(X_tr, y_tr)

    val_prob = pipe.predict_proba(X_val)[:, 1]
    oof_probs[val_idx] = val_prob
    test_probs += pipe.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y_val, val_prob)
    print(f"Fold {fold}/{n_splits} - ROC-AUC: {fold_auc:.5f}")

# 3. Đánh giá tổng hợp Out-Of-Fold (OOF)
oof_auc = roc_auc_score(y, oof_probs)
print(f"\nOverall OOF ROC-AUC: {oof_auc:.5f}")
print(classification_report(y, (oof_probs >= 0.5).astype(int)))

# 4. Xuất file nộp Kaggle (trung bình xác suất qua 5 folds)
submission = pd.DataFrame({
    "id": test["id"],
    "is_fraud": test_probs,
})
submission.to_csv("submission.csv", index=False)
print("Đã lưu submission.csv:", submission.shape)
