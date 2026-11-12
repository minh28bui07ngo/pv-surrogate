import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# CẤU HÌNH
# ==========================================================
N_TRAIN = 4000          # số mẫu huấn luyện dùng cho Random Forest
SEED = 1
N_ITER_SEARCH = 30     # số tổ hợp siêu tham số thử trong RandomizedSearchCV
CV_FOLDS = 5
POP_SIZE = 6400
N_GEN = 500

OUTPUT_NAMES = [
    "Electricity:Facility",
    "ElectricityProduced:Facility",
    "E_self",
    "Annual_grid_cost(VND)",
]

# ==========================================================
# BƯỚC 1: ĐỌC DỮ LIỆU VÀ LẤY N_TRAIN MẪU
# ==========================================================
INPUT_PKL  = "D:/da2/5000/inputs5000.pkl"    # <-- đổi thành tên file input tổng của bạn
OUTPUT_PKL = "D:/da2/5000/outputs5000.pkl"   # <-- đổi thành tên file output tổng của bạn

TEST_FRACTION = 0.20          # 20% test, 80% train
SPLIT_SEED    = 2000            # cố định seed để mỗi lần chạy chia giống nhau
STRATIFY_BY_TOTAL_PV = True   # chia đều theo tổng công suất PV (xem split_indices)

inputs_full  = pd.read_pickle(INPUT_PKL)
outputs_full = pd.read_pickle(OUTPUT_PKL)

# Dòng i của input phải ứng với dòng i của output
assert len(inputs_full) == len(outputs_full), (
    f"Số mẫu input ({len(inputs_full)}) khác số mẫu output ({len(outputs_full)})"
)
inputs_full  = inputs_full.reset_index(drop=True)
outputs_full = outputs_full.reset_index(drop=True)

def split_indices(total_pv, test_fraction, seed, stratify=True, n_bins=10):
    """
    Trả về (train_idx, test_idx): vị trí dòng, không trùng nhau, cộng lại đủ mọi mẫu.

    stratify=True: chia mẫu thành n_bins nhóm theo tổng công suất PV rồi lấy đúng
    test_fraction trong MỖI nhóm, để train và test cùng phủ đủ dải công suất
    (kể cả vùng PV nhỏ vốn ít mẫu).
    """
    rng = np.random.default_rng(seed)
    n = len(total_pv)
    if stratify:
        bins = pd.qcut(pd.Series(total_pv), q=n_bins,
                       labels=False, duplicates="drop").to_numpy()
    else:
        bins = np.zeros(n, dtype=int)

    train_parts, test_parts = [], []
    for b in np.unique(bins):
        idx = rng.permutation(np.flatnonzero(bins == b))
        n_test = int(round(len(idx) * test_fraction))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    # Xáo trộn lại: Keras validation_split lấy 15% CUỐI của dữ liệu train
    # (không tự xáo), nên thứ tự phải ngẫu nhiên.
    train_idx = rng.permutation(np.concatenate(train_parts))
    test_idx  = rng.permutation(np.concatenate(test_parts))
    return train_idx, test_idx

# Tổng công suất 8 PV (cùng các cột 32:40 mà phần sau dùng làm input)
total_pv_all = inputs_full.iloc[:, 32:40].sum(axis=1).to_numpy(dtype=float)
train_idx, test_idx = split_indices(
    total_pv_all, TEST_FRACTION, SPLIT_SEED, stratify=STRATIFY_BY_TOTAL_PV
)
assert len(np.intersect1d(train_idx, test_idx)) == 0
assert len(train_idx) + len(test_idx) == len(inputs_full)

# reset_index(drop=True): phần sau ghép input với output theo chỉ số dòng 0..n-1
inputs_all          = inputs_full.iloc[train_idx].reset_index(drop=True)
outputs_hourly      = outputs_full.iloc[train_idx].reset_index(drop=True)
inputs_all_test     = inputs_full.iloc[test_idx].reset_index(drop=True)
outputs_hourly_test = outputs_full.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(inputs_all)} mẫu | Test: {len(inputs_all_test)} mẫu "
      f"({len(inputs_all_test)/len(inputs_full):.0%})")

rng = np.random.default_rng(SEED)
if N_TRAIN < len(inputs_all):
    idx = np.sort(rng.choice(len(inputs_all), size=N_TRAIN, replace=False))
    inputs_all = inputs_all.iloc[idx].reset_index(drop=True)
    outputs_hourly = outputs_hourly.iloc[idx].reset_index(drop=True)
else:
    inputs_all = inputs_all.reset_index(drop=True)
    outputs_hourly = outputs_hourly.reset_index(drop=True)
inputs_all_test = inputs_all_test.reset_index(drop=True)
outputs_hourly_test = outputs_hourly_test.reset_index(drop=True)

print(f"Số mẫu train: {len(inputs_all)} | Số mẫu test: {len(inputs_all_test)}")
print("Done step 1")


# ==========================================================
# BƯỚC 2: BIỂU GIÁ ĐIỆN VÀ OUTPUT TỔNG NĂM
# ==========================================================
def get_electricity_rate(hour):
    # Giờ thấp điểm: 22:00 đến trước 04:00
    if (hour >= 22) or (hour < 4):
        return 1918
    # Giờ cao điểm
    elif ((hour >= 9.5) and (hour < 11.5)) or ((hour >= 13.5) and (hour < 17)):
        return 3152
    # Giờ bình thường
    else:
        return 5422


def compute_annual_outputs(outputs_hourly_df):
    results = []
    for _, row in outputs_hourly_df.iterrows():
        load = np.asarray(row["Electricity:Facility"], dtype=float)
        pv = np.asarray(row["ElectricityProduced:Facility"], dtype=float)

        if len(load) != len(pv):
            raise ValueError(
                "Electricity:Facility và ElectricityProduced:Facility "
                "không cùng độ dài."
            )

        e_self_hourly = np.minimum(load, pv)
        grid_power_hourly = np.maximum(load - pv, 0)

        # Giữ nguyên hệ số quy đổi của chương trình gốc
        grid_energy_hourly_kWh = grid_power_hourly / (1000 * 3600)

        hours = np.arange(len(load)) % 24
        rate_hourly = np.array(
            [get_electricity_rate(h) for h in hours], dtype=float
        )
        annual_grid_cost = (grid_energy_hourly_kWh * rate_hourly).sum()

        results.append([load.sum(), pv.sum(), e_self_hourly.sum(), annual_grid_cost])

    return pd.DataFrame(results, columns=OUTPUT_NAMES)


annual_outputs_test = compute_annual_outputs(outputs_hourly_test)
annual_outputs = compute_annual_outputs(outputs_hourly)
print("Done step 2")

inputs_test = inputs_all_test.iloc[:, 32:40]
inputs = inputs_all.iloc[:, 32:40]

dataset_test = inputs_test.join(annual_outputs_test)
dataset = inputs.join(annual_outputs)
print(dataset.head(10))
print("Done step 3")

# ==========================================================
# BƯỚC 3: CHUẨN BỊ DỮ LIỆU
# Random Forest chia nhánh theo ngưỡng nên KHÔNG cần chuẩn hóa
# input hay output (bất biến với phép biến đổi đơn điệu).
# Dùng mảng NumPy để tránh cảnh báo tên đặc trưng của sklearn.
# ==========================================================
X_train = dataset[inputs.columns].to_numpy(dtype=float)
X_test = dataset_test[inputs_test.columns].to_numpy(dtype=float)
Y_train = dataset[OUTPUT_NAMES].to_numpy(dtype=float)
Y_test = dataset_test[OUTPUT_NAMES].to_numpy(dtype=float)
print("Done step 4")


# ==========================================================
# BƯỚC 4: HUẤN LUYỆN RANDOM FOREST (1 RỪNG CHO MỖI OUTPUT)
# Siêu tham số được chọn bằng RandomizedSearchCV (5-fold CV)
# ==========================================================
param_dist = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 5, 8, 12],
    "min_samples_leaf": [1, 2, 3, 5],
    "min_samples_split": [2, 4, 8],
    "max_features": [1.0, 0.75, 0.5, "sqrt"],
    "max_samples": [None, 0.7, 0.85],
}

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

t1 = time.time()
rfs = []
best_params_all = []
for j, name in enumerate(OUTPUT_NAMES):
    search = RandomizedSearchCV(
        RandomForestRegressor(bootstrap=True, random_state=SEED, n_jobs=1),
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, Y_train[:, j])
    best = search.best_params_

    # Huấn luyện lại với tham số tốt nhất, bật OOB để có ước lượng độc lập
    rf = RandomForestRegressor(
        **best, bootstrap=True, oob_score=True, random_state=SEED, n_jobs=-1
    ).fit(X_train, Y_train[:, j])
    rf.set_params(n_jobs=1)  # batch nhỏ trong NSGA-II chạy nhanh hơn khi n_jobs=1

    rfs.append(rf)
    best_params_all.append(best)
    print(f"\n{name}")
    print(f"  best params : {best}")
    print(f"  OOB R2      : {rf.oob_score_:.4f}")

print(f"\nThời gian huấn luyện Random Forest: {(time.time() - t1) / 60:.2f} phút")
joblib.dump(
    {"rfs": rfs, "output_names": OUTPUT_NAMES, "best_params": best_params_all},
    "best_rf_surrogate.joblib",
)
print("Done step 5")


def predict_all(X, return_std=False):
    """Dự đoán 4 output; return_std=True trả thêm độ lệch chuẩn giữa các cây."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if not return_std:
        return np.column_stack([rf.predict(X) for rf in rfs])
    means, stds = [], []
    for rf in rfs:
        tree_pred = np.stack([t.predict(X) for t in rf.estimators_])
        means.append(tree_pred.mean(axis=0))
        stds.append(tree_pred.std(axis=0))
    return np.column_stack(means), np.column_stack(stds)


def Prediction(population_PV):
    return predict_all(population_PV)


# ==========================================================
# BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH
# ==========================================================
def safe_div(a, b):
    return a / np.maximum(b, 1.0)


def report_metrics(title, y_true, y_pred):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    rows = []
    for i, name in enumerate(OUTPUT_NAMES):
        t, p = y_true[:, i], y_pred[:, i]
        r2 = r2_score(t, p)
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        nrmse_mean = rmse / abs(t.mean()) if t.mean() != 0 else np.nan
        nrmse_range = rmse / (t.max() - t.min()) if t.max() > t.min() else np.nan
        rows.append([name, r2, mae, rmse, nrmse_mean * 100, nrmse_range * 100,
                     np.max(np.abs(p - t))])
        print(f"\n{name}")
        print(f"R2           : {r2:.6f}")
        print(f"MAE          : {mae:.6e}")
        print(f"RMSE         : {rmse:.6e}")
        print(f"NRMSE(mean)  : {nrmse_mean:.4%}")
        print(f"NRMSE(range) : {nrmse_range:.4%}")
        print(f"Max error    : {np.max(np.abs(p - t)):.6e}")
    return pd.DataFrame(
        rows,
        columns=["Output", "R2", "MAE", "RMSE", "NRMSE_mean_%", "NRMSE_range_%", "MaxErr"],
    )


def physical_check(title, y_pred):
    facility, produced, e_self = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    SS, SC = safe_div(e_self, facility), safe_div(e_self, produced)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Tổng số mẫu: {len(y_pred)}")
    print(f"SS > 1: {np.sum(SS > 1)} ({100 * np.mean(SS > 1):.2f}%)")
    print(f"SC > 1: {np.sum(SC > 1)} ({100 * np.mean(SC > 1):.2f}%)")
    print(f"Grid cost < 0: {np.sum(y_pred[:, 3] < 0)}")
    print(f"SS: {SS.min():.6f} -> {SS.max():.6f}")
    print(f"SC: {SC.min():.6f} -> {SC.max():.6f}")


# 5a. Cross-validation trên tập train với siêu tham số đã chọn
cv_pred = np.column_stack(
    [
        cross_val_predict(
            RandomForestRegressor(**best_params_all[j], bootstrap=True,
                                  random_state=SEED, n_jobs=-1),
            X_train, Y_train[:, j], cv=cv,
        )
        for j in range(len(OUTPUT_NAMES))
    ]
)
report_metrics(f"{CV_FOLDS}-FOLD CROSS-VALIDATION ON TRAINING SET", Y_train, cv_pred)

# 5b. Out-of-bag (ước lượng độc lập, không cần tập validation riêng)
oob_pred = np.column_stack([rf.oob_prediction_ for rf in rfs])
report_metrics("OUT-OF-BAG PREDICTION ON TRAINING SET", Y_train, oob_pred)

# 5c. Trên chính tập train (thường lạc quan hơn thực tế)
y_train_pred = predict_all(X_train)
report_metrics("PERFORMANCE ON TRAINING SET (optimistic)", Y_train, y_train_pred)
physical_check("PHYSICAL CONSTRAINT CHECK - TRAIN", y_train_pred)

# 5d. Tập test
y_test_pred, y_test_std = predict_all(X_test, return_std=True)
summary_test = report_metrics("PERFORMANCE ON TEST SET", Y_test, y_test_pred)
physical_check("PHYSICAL CONSTRAINT CHECK - TEST", y_test_pred)
print("\n", summary_test)

# 5e. Độ phân tán giữa các cây trên tập test
print("\nĐộ lệch chuẩn giữa các cây trung bình (đơn vị gốc):")
for i, name in enumerate(OUTPUT_NAMES):
    print(f"  {name}: {np.mean(y_test_std[:, i]):.6e}")

# 5f. Độ quan trọng của từng vùng PV (thay cho độ dài tương quan ARD của Kriging)
imp = pd.DataFrame(
    {name: rf.feature_importances_ for name, rf in zip(OUTPUT_NAMES, rfs)},
    index=inputs.columns,
)
print("\nFeature importance (mean decrease in impurity):")
print(imp.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
imp.plot.bar(ax=ax)
ax.set_ylabel("Importance")
ax.set_title("Random Forest feature importance per output")
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.show()

# 5g. Parity plot
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for i, ax in enumerate(axes.ravel()):
    ax.scatter(Y_test[:, i], y_test_pred[:, i], s=20, edgecolors="black", linewidths=0.3)
    lim = [min(Y_test[:, i].min(), y_test_pred[:, i].min()),
           max(Y_test[:, i].max(), y_test_pred[:, i].max())]
    ax.plot(lim, lim, "--", linewidth=1)
    ax.set_title(OUTPUT_NAMES[i])
    ax.set_xlabel("True")
    ax.set_ylabel("Random Forest prediction")
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()
print("Done step 6")


# ==========================================================
# BƯỚC 6: LCC, SS, SC
# ==========================================================
def calculate_LCC_SS_SC(population_PV, time=20, discount_rate=0.1):
    # Công suất định mức -> diện tích mỗi tấm
    area_PV = population_PV / 198
    total_area_PV = np.sum(area_PV, axis=1)

    # Chi phí đầu tư ban đầu
    initial_PV_cost = total_area_PV * 2_500_000 * 1.15

    output_pred = Prediction(population_PV)
    annual_grid_cost = output_pred[:, 3]

    # Hiện giá tiền điện mua từ lưới trong 20 năm
    present_value_grid_cost = np.zeros(len(population_PV), dtype=float)
    for year in range(1, time + 1):
        present_value_grid_cost += annual_grid_cost / (1 + discount_rate) ** year

    LCC = initial_PV_cost + present_value_grid_cost
    SS = safe_div(output_pred[:, 2], output_pred[:, 0])
    SC = safe_div(output_pred[:, 2], output_pred[:, 1])
    return LCC, SS, SC, annual_grid_cost


# ==========================================================
# BƯỚC 6b: SO SÁNH SAI SỐ CỦA 3 TIÊU CHÍ (LCC, SS, SC)
#   A) Gián tiếp : RF huấn luyện cho 4 output -> tính LCC/SS/SC từ dự đoán
#   B) Trực tiếp : RF huấn luyện thẳng cho LCC/SS/SC
# Cả hai đều được so với tiêu chí tính từ Y_test (dữ liệu thật).
#
# DÁN KHỐI NÀY VÀO ngay sau dòng  print("Done step 6")  và trước BƯỚC 7.
# Dùng lại các biến có sẵn: X_train, X_test, Y_train, Y_test, rfs,
# param_dist, cv, N_ITER_SEARCH, SEED, predict_all, safe_div.
# ==========================================================
CRITERIA_NAMES = ["LCC", "SS", "SC"]
 
 
def criteria_from_outputs(X, Y, years=20, discount_rate=0.1):
    """Cùng công thức với calculate_LCC_SS_SC nhưng nhận sẵn 4 output Y
    (thật hoặc dự đoán) thay vì tự gọi mô hình. Trả về mảng (n, 3): LCC, SS, SC."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
 
    initial_PV_cost = (X / 198).sum(axis=1) * 2_500_000 * 1.15
    annuity = sum(1.0 / (1 + discount_rate) ** t for t in range(1, years + 1))
    LCC = initial_PV_cost + Y[:, 3] * annuity
    SS = safe_div(Y[:, 2], Y[:, 0])
    SC = safe_div(Y[:, 2], Y[:, 1])
    return np.column_stack([LCC, SS, SC])
 
 
# Kiểm tra hàm mới khớp hàm gốc (để chắc chắn so sánh công bằng)
_ref = np.column_stack(calculate_LCC_SS_SC(X_test)[:3])
assert np.allclose(_ref, criteria_from_outputs(X_test, predict_all(X_test)), rtol=1e-9), \
    "criteria_from_outputs không khớp calculate_LCC_SS_SC"
 
# --- Giá trị thật của tiêu chí (từ dữ liệu test và train) ---
C_test_true = criteria_from_outputs(X_test, Y_test)
C_train_true = criteria_from_outputs(X_train, Y_train)
 
# --- A) Gián tiếp: tiêu chí từ dự đoán 4 output ---
C_test_indirect = criteria_from_outputs(X_test, y_test_pred)
 
# --- B) Trực tiếp: huấn luyện 3 rừng riêng cho LCC, SS, SC ---
t2 = time.time()
rfs_direct, best_params_direct = [], []
for j, name in enumerate(CRITERIA_NAMES):
    search = RandomizedSearchCV(
        RandomForestRegressor(bootstrap=True, random_state=SEED, n_jobs=1),
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, C_train_true[:, j])
    best = search.best_params_
    rf = RandomForestRegressor(
        **best, bootstrap=True, oob_score=True, random_state=SEED, n_jobs=-1
    ).fit(X_train, C_train_true[:, j])
    rfs_direct.append(rf)
    best_params_direct.append(best)
    print(f"\n[Direct] {name}: OOB R2 = {rf.oob_score_:.4f} | {best}")
print(f"\nThời gian huấn luyện mô hình trực tiếp: {(time.time() - t2) / 60:.2f} phút")
 
C_test_direct = np.column_stack([rf.predict(X_test) for rf in rfs_direct])
 
 
# --- Chỉ số sai số ---
def criteria_metrics(y_true, y_pred):
    rows = []
    for i, name in enumerate(CRITERIA_NAMES):
        t, p = y_true[:, i], y_pred[:, i]
        rmse = np.sqrt(mean_squared_error(t, p))
        rows.append({
            "Criterion": name,
            "R2": r2_score(t, p),
            "MAE": mean_absolute_error(t, p),
            "RMSE": rmse,
            "NRMSE_mean_%": 100 * rmse / abs(t.mean()) if t.mean() != 0 else np.nan,
            "NRMSE_range_%": 100 * rmse / (t.max() - t.min()) if t.max() > t.min() else np.nan,
            "MaxErr": np.max(np.abs(p - t)),
        })
    return pd.DataFrame(rows).set_index("Criterion")
 
 
m_indirect = criteria_metrics(C_test_true, C_test_indirect)
m_direct = criteria_metrics(C_test_true, C_test_direct)
 
comparison = pd.concat(
    {"A: gián tiếp (4 output -> tiêu chí)": m_indirect,
     "B: trực tiếp (huấn luyện thẳng)": m_direct},
    axis=1,
)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n" + "=" * 80)
print("SO SÁNH SAI SỐ TIÊU CHÍ TRÊN TẬP TEST")
print("=" * 80)
print(comparison.to_string(float_format=lambda v: f"{v:.6g}"))
comparison.to_csv("criteria_error_comparison.csv")
 
# Tỉ lệ RMSE (B / A): < 1 nghĩa là huấn luyện trực tiếp tốt hơn
ratio = (m_direct["RMSE"] / m_indirect["RMSE"]).rename("RMSE_direct / RMSE_indirect")
print("\n", ratio.to_string(float_format=lambda v: f"{v:.3f}"))
 
# --- Parity plot: hai cách trên cùng một hình cho mỗi tiêu chí ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for i, (ax, name) in enumerate(zip(axes, CRITERIA_NAMES)):
    t = C_test_true[:, i]
    ax.scatter(t, C_test_indirect[:, i], s=14, alpha=0.6, label="A: gián tiếp")
    ax.scatter(t, C_test_direct[:, i], s=14, alpha=0.6, marker="x", label="B: trực tiếp")
    lim = [t.min(), t.max()]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_title(f"{name}  (RMSE A={m_indirect.loc[name, 'RMSE']:.3g}, "
                 f"B={m_direct.loc[name, 'RMSE']:.3g})", fontsize=10)
    ax.set_xlabel("True (từ Y_test)")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("Done step 6b")


# ==========================================================
# BƯỚC 7: NSGA-II (RÀNG BUỘC VẬT LÝ ĐƯA VÀO TRỰC TIẾP)
# ==========================================================
lower_bounds = np.array([
    198.000621, 198.000621, 197.999648, 197.999648,
    990.000337, 990.000337, 198.001915, 198.001915,
])
upper_bounds = np.array([
    73656.000079, 73656.000079, 36827.998902, 36827.998902,
    96227.999467, 96227.999467, 69695.998978, 69695.998978,
])


class PVProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=8,
            n_obj=3,
            n_ieq_constr=3,   # SS <= 1, SC <= 1, grid cost >= 0
            xl=lower_bounds,
            xu=upper_bounds,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        LCC, SS, SC, grid_cost = calculate_LCC_SS_SC(X)
        out["F"] = np.column_stack([-SS, -SC, LCC])
        out["G"] = np.column_stack([SS - 1.0, SC - 1.0, -grid_cost])


problem = PVProblem()
algorithm = NSGA2(pop_size=POP_SIZE, eliminate_duplicates=True)
termination = get_termination("n_gen", N_GEN)

result = minimize(problem, algorithm, termination, seed=SEED, verbose=True, save_history=True)

if result.X is None:
    raise RuntimeError(
        "NSGA-II không tìm được nghiệm thỏa mãn ràng buộc. "
        "Hãy kiểm tra chất lượng mô hình Random Forest hoặc nới ràng buộc."
    )
print("Done step 7")

# ==========================================================
# BƯỚC 8: KẾT QUẢ PARETO
# ==========================================================
X_pareto = np.atleast_2d(result.X).astype(float)
F_pareto = np.atleast_2d(result.F).astype(float)

SS_pareto = -F_pareto[:, 0]
SC_pareto = -F_pareto[:, 1]
LCC_pareto = F_pareto[:, 2]

print("Số nghiệm Pareto (khả thi):", len(X_pareto))
print(f"SS : {SS_pareto.min():.6f} -> {SS_pareto.max():.6f}")
print(f"SC : {SC_pareto.min():.6f} -> {SC_pareto.max():.6f}")
print(f"LCC: {LCC_pareto.min():,.2f} -> {LCC_pareto.max():,.2f}")

# Các front bị chi phối (chỉ lấy cá thể khả thi trong quần thể cuối)
F_pop = np.asarray(result.pop.get("F"), dtype=float)
feasible = np.asarray(result.pop.get("feasible")).reshape(-1)
F_pop = F_pop[feasible] if feasible.any() else F_pop
SS_pop, SC_pop, LCC_pop = -F_pop[:, 0], -F_pop[:, 1], F_pop[:, 2]
front_indices = NonDominatedSorting().do(F_pop)
print("Số front trong quần thể cuối:", len(front_indices))

# Nghiệm đặc trưng trên front 1 (result.F)
idx_max_ss = int(np.argmax(SS_pareto))
idx_max_sc = int(np.argmax(SC_pareto))
idx_min_lcc = int(np.argmin(LCC_pareto))

F_range = F_pareto.max(axis=0) - F_pareto.min(axis=0)
F_range[F_range == 0] = 1.0
F_norm = (F_pareto - F_pareto.min(axis=0)) / F_range
idx_knee = int(np.argmin(np.linalg.norm(F_norm, axis=1)))

special = [
    (idx_max_ss, "*", 150, "Maximum SS"),
    (idx_max_sc, "s", 80, "Maximum SC"),
    (idx_min_lcc, "D", 75, "Minimum LCC"),
    (idx_knee, "X", 110, "Knee point"),
]

# ==========================================================
# BƯỚC 9: VẼ ĐỒ THỊ
# ==========================================================
# 3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for rank in range(min(len(front_indices), 3) - 1, 0, -1):
    ii = np.asarray(front_indices[rank], dtype=int)
    ax.scatter(SS_pop[ii], SC_pop[ii], LCC_pop[ii], s=15, alpha=0.25, edgecolors="none")
sc3d = ax.scatter(SS_pareto, SC_pareto, LCC_pareto, c=LCC_pareto, s=34,
                  alpha=0.9, edgecolors="none", label="Pareto Front 1")
for i, m, s, lab in special:
    ax.scatter(SS_pareto[i], SC_pareto[i], LCC_pareto[i], marker=m, s=s,
               edgecolors="black", linewidths=0.8, label=lab)
ax.set_title("Three-Dimensional Pareto-Optimal Solutions (Random Forest)", fontsize=14, pad=18)
ax.set_xlabel("Self-Sufficiency (SS)")
ax.set_ylabel("Self-Consumption (SC)")
ax.set_zlabel("Life-Cycle Cost (LCC)")
fig.colorbar(sc3d, ax=ax, shrink=0.72, pad=0.10).set_label("LCC")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


def projection(x_all, y_all, x1, y1, c1, xl, yl, cl, title, hlines=(), vlines=()):
    plt.figure(figsize=(8, 6))
    for rank in range(min(len(front_indices), 3) - 1, 0, -1):
        ii = np.asarray(front_indices[rank], dtype=int)
        plt.scatter(x_all[ii], y_all[ii], s=15, alpha=0.25, edgecolors="none",
                    label=f"Front {rank + 1}")
    pr = plt.scatter(x1, y1, c=c1, s=34, alpha=0.95, edgecolors="none",
                     label="Pareto Front 1", zorder=5)
    for i, m, s, lab in special:
        plt.scatter(x1[i], y1[i], marker=m, s=s, edgecolors="black",
                    linewidths=0.8, label=lab, zorder=10)
    for h in hlines:
        plt.axhline(h, linestyle="--", linewidth=1.1, label=f"Physical limit = {h}")
    for v in vlines:
        plt.axvline(v, linestyle="--", linewidth=1.1)
    plt.title(title, fontsize=13)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.colorbar(pr).set_label(cl)
    plt.grid(alpha=0.15)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


projection(LCC_pop, SS_pop, LCC_pareto, SS_pareto, SC_pareto,
           "Life-Cycle Cost (LCC)", "Self-Sufficiency (SS)", "Self-Consumption (SC)",
           "Projection: LCC versus SS")
projection(LCC_pop, SC_pop, LCC_pareto, SC_pareto, SS_pareto,
           "Life-Cycle Cost (LCC)", "Self-Consumption (SC)", "Self-Sufficiency (SS)",
           "Projection: LCC versus SC", hlines=(1.0,))
projection(SS_pop, SC_pop, SS_pareto, SC_pareto, LCC_pareto,
           "Self-Sufficiency (SS)", "Self-Consumption (SC)", "Life-Cycle Cost (LCC)",
           "Projection: SS versus SC", hlines=(1.0,), vlines=(1.0,))

# ==========================================================
# BƯỚC 10: THIẾT KẾ ĐẶC TRƯNG + ĐỘ TIN CẬY CỦA RANDOM FOREST
# Random Forest không ngoại suy được: kiểm tra khoảng cách từ nghiệm
# tối ưu tới mẫu train gần nhất (đơn vị: độ lệch chuẩn của từng biến).
# ==========================================================
x_sd = X_train.std(axis=0, ddof=1)
x_sd[x_sd == 0] = 1.0


def nearest_train_distance(x):
    d = np.linalg.norm((X_train - x) / x_sd, axis=1)
    return d.min()


# Khoảng cách láng giềng gần nhất điển hình giữa các mẫu train (leave-one-out)
typical_nn = np.median([
    np.min(np.delete(np.linalg.norm((X_train - X_train[i]) / x_sd, axis=1), i))
    for i in range(len(X_train))
])
print(f"\nKhoảng cách láng giềng gần nhất điển hình giữa các mẫu train: {typical_nn:.3f}")

representative_designs = {
    "Maximum SS": X_pareto[idx_max_ss],
    "Maximum SC": X_pareto[idx_max_sc],
    "Minimum LCC": X_pareto[idx_min_lcc],
    "Knee point": X_pareto[idx_knee],
}

print("\nBiến thiết kế của các nghiệm đặc trưng")
for name, x in representative_designs.items():
    mean_p, std_p = predict_all(x.reshape(1, -1), return_std=True)
    rel_unc = std_p[0] / np.maximum(np.abs(mean_p[0]), 1.0)
    d = nearest_train_distance(x)
    print(f"\n{name}:")
    print(x)
    print("Độ phân tán giữa các cây, tương đối (Facility, Produced, E_self, Cost):")
    print(np.round(rel_unc * 100, 4), "%")
    print(f"Khoảng cách tới mẫu train gần nhất: {d:.3f} "
          f"({d / typical_nn:.2f}x điển hình)"
          + ("  <-- xa dữ liệu train, nên kiểm chứng lại bằng EnergyPlus" if d > 2 * typical_nn else ""))

# Lưu để kiểm chứng lại bằng EnergyPlus
pd.DataFrame(X_pareto, columns=inputs.columns).assign(
    SS=SS_pareto, SC=SC_pareto, LCC=LCC_pareto
).to_csv("pareto_solutions_rf.csv", index=False)
print("Done step 8")
