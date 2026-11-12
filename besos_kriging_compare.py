import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================================
# CẤU HÌNH
# ==========================================================
N_TRAIN = 4000          # số mẫu huấn luyện dùng cho Kriging
SEED = 1
USE_LOG_INPUT = False  # True: lấy log diện tích/công suất PV trước khi chuẩn hóa
N_RESTARTS = 10        # số lần khởi tạo lại khi tối ưu siêu tham số kernel
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
SPLIT_SEED    = 42            # cố định seed để mỗi lần chạy chia giống nhau
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
# BƯỚC 3: CHUẨN HÓA INPUT VÀ OUTPUT
# ==========================================================
def transform_inputs(x):
    x = np.asarray(x, dtype=float)
    return np.log(x) if USE_LOG_INPUT else x


X_train_tf = transform_inputs(dataset[inputs.columns].to_numpy())
X_test_tf = transform_inputs(dataset_test[inputs_test.columns].to_numpy())

input_mean = X_train_tf.mean(axis=0)
input_std = X_train_tf.std(axis=0, ddof=1)
input_std[input_std == 0] = 1.0


def Standardization(population_PV):
    return (transform_inputs(population_PV) - input_mean) / input_std


normed_train_data = Standardization(dataset[inputs.columns].to_numpy())
normed_test_data = Standardization(dataset_test[inputs_test.columns].to_numpy())

Y_train = dataset[OUTPUT_NAMES].to_numpy(dtype=float)
Y_test = dataset_test[OUTPUT_NAMES].to_numpy(dtype=float)

output_mean = Y_train.mean(axis=0)
output_std = Y_train.std(axis=0, ddof=1)
output_std[output_std == 0] = 1.0


def norm_output(y):
    return (np.asarray(y, dtype=float) - output_mean) / output_std


def unnorm_output(y):
    return np.asarray(y, dtype=float) * output_std + output_mean


train_labels = norm_output(Y_train)
test_labels = norm_output(Y_test)
print("Done step 4")

# ==========================================================
# SO SÁNH SAI SỐ CỦA 3 TIÊU CHÍ (LCC, SS, SC) - KRIGING, DÙNG MÔ HÌNH ĐÃ LƯU
#   A) Gián tiếp : Kriging 4 output (Facility, Produced, E_self, Cost)
#                  -> tính LCC/SS/SC từ dự đoán
#   B) Trực tiếp : Kriging huấn luyện thẳng cho [LCC, SS, SC]
# Cả hai được so với tiêu chí tính từ Y_test (dữ liệu thật).
#
# DÁN KHỐI NÀY ngay sau  print("Done step 4")  (cuối phần chuẩn hóa dữ liệu,
# TRƯỚC "BƯỚC 4: HUẤN LUYỆN KRIGING"). Khối này KHÔNG cần huấn luyện lại: nó nạp
# cả hai mô hình từ file .joblib. Nếu chỉ cần phần so sánh, bạn có thể chạy
# một bản sao của chương trình dừng ngay sau khối này.
#
# Cần có sẵn từ các bước trước: dataset, dataset_test, inputs, inputs_test,
# Y_train, Y_test, OUTPUT_NAMES, và các import (np, pd, plt, joblib, r2_score,
# mean_absolute_error, mean_squared_error).
#
# Lưu ý quan trọng: hai file mô hình phải được huấn luyện trên ĐÚNG tập train
# này (cùng SPLIT_SEED, SEED, N_TRAIN, TEST_FRACTION). Nếu không, tập test
# hiện tại có thể đã nằm trong dữ liệu huấn luyện của một trong hai mô hình.
# ==========================================================
import itertools
 
INDIRECT_MODEL_PATH = "D:/da2/kriging_3O/4_best_kriging_surrogate.joblib"          # 4 output
DIRECT_MODEL_PATH = "D:/da2/kriging_3O/3_best_kriging_surrogate.joblib"     # <-- đổi: 3 tiêu chí [LCC, SS, SC]
CRITERIA_NAMES = ["LCC", "SS", "SC"]
STRICT_CHECK = True   # True: dừng nếu file mô hình không khớp tập train hiện tại
 
 
def safe_div(a, b):
    return a / np.maximum(b, 1.0)
 
 
def criteria_from_outputs(X, Y, years=20, discount_rate=0.1):
    """Cùng công thức với calculate_LCC_SS_SC nhưng nhận sẵn 4 output Y
    (thật hoặc dự đoán). Trả về mảng (n, 3): LCC, SS, SC."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
 
    initial_PV_cost = (X / 198).sum(axis=1) * 2_500_000 * 1.15
    annuity = sum(1.0 / (1 + discount_rate) ** t for t in range(1, years + 1))
    LCC = initial_PV_cost + Y[:, 3] * annuity
    SS = safe_div(Y[:, 2], Y[:, 0])
    SC = safe_div(Y[:, 2], Y[:, 1])
    return np.column_stack([LCC, SS, SC])
 
 
# --- Dữ liệu thô (đơn vị gốc) ---
X_train_raw = dataset[inputs.columns].to_numpy(dtype=float)
X_test_raw = dataset_test[inputs_test.columns].to_numpy(dtype=float)
 
C_train_true = criteria_from_outputs(X_train_raw, Y_train)
C_test_true = criteria_from_outputs(X_test_raw, Y_test)
 
 
# --- Nạp và dùng mô hình Kriging đã lưu ---
def load_kriging(path):
    bundle = joblib.load(path)
    for key in ("gps", "input_mean", "input_std", "output_mean", "output_std"):
        if key not in bundle:
            raise KeyError(f"{path}: thiếu khóa '{key}' (cần cùng định dạng file với chương trình Kriging)")
    return bundle
 
 
def predict_kriging(bundle, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_tf = np.log(X) if bundle.get("use_log_input", False) else X
    X_norm = (X_tf - bundle["input_mean"]) / bundle["input_std"]
    y_norm = np.column_stack([gp.predict(X_norm) for gp in bundle["gps"]])
    return y_norm * bundle["output_std"] + bundle["output_mean"]
 
 
def check_bundle(bundle, label, n_outputs, y_train_ref):
    """Kiểm tra file mô hình khớp với tập train hiện tại (và đúng thứ tự output)."""
    problems = []
    if len(bundle["gps"]) != n_outputs:
        problems.append(f"có {len(bundle['gps'])} GP, cần {n_outputs}")
    X_tf = np.log(X_train_raw) if bundle.get("use_log_input", False) else X_train_raw
    if not np.allclose(bundle["input_mean"], X_tf.mean(axis=0), rtol=1e-6):
        problems.append("input_mean khác trung bình input của tập train hiện tại")
    if not np.allclose(bundle["output_mean"], y_train_ref.mean(axis=0), rtol=1e-6):
        problems.append("output_mean khác trung bình output train hiện tại "
                        "(sai thứ tự output, sai công thức/đơn vị tiêu chí, hoặc sai tập train)"
                        f"\n    lưu trong file : {np.asarray(bundle['output_mean'])}"
                        f"\n    tính từ train  : {y_train_ref.mean(axis=0)}")
    if problems:
        msg = f"[{label}] " + "; ".join(problems)
        if STRICT_CHECK:
            raise AssertionError(msg)
        print("CẢNH BÁO:", msg)
    else:
        print(f"[{label}] khớp tập train hiện tại.")
 
 
indirect_model = load_kriging(INDIRECT_MODEL_PATH)
direct_model = load_kriging(DIRECT_MODEL_PATH)
check_bundle(indirect_model, "Indirect", len(OUTPUT_NAMES), Y_train)
 
 
def detect_criteria_order(bundle):
    """Tìm thứ tự các tiêu chí mà mô hình trực tiếp đã lưu, bằng cách khớp
    output_mean với trung bình [LCC, SS, SC] của tập train theo mọi hoán vị.
    Trả về perm sao cho cột k của mô hình ứng với tiêu chí perm[k]; None nếu không khớp."""
    for perm in itertools.permutations(range(len(CRITERIA_NAMES))):
        if np.allclose(bundle["output_mean"], C_train_true[:, list(perm)].mean(axis=0), rtol=1e-6):
            return list(perm)
    return None
 
 
direct_order = detect_criteria_order(direct_model)
if direct_order is None:
    # Không hoán vị nào khớp: để check_bundle báo lỗi kèm giá trị cụ thể
    check_bundle(direct_model, "Direct", len(CRITERIA_NAMES), C_train_true)
    direct_order = list(range(len(CRITERIA_NAMES)))
else:
    print("[Direct] thứ tự output trong file:", [CRITERIA_NAMES[k] for k in direct_order],
          "-> sẽ được sắp lại thành", CRITERIA_NAMES)
    check_bundle(direct_model, "Direct", len(CRITERIA_NAMES), C_train_true[:, direct_order])
 
# A) Gián tiếp: dự đoán 4 output -> tiêu chí
Y_test_pred_indirect = predict_kriging(indirect_model, X_test_raw)
C_test_indirect = criteria_from_outputs(X_test_raw, Y_test_pred_indirect)
 
# B) Trực tiếp: dự đoán thẳng [LCC, SS, SC]
C_test_direct_raw = predict_kriging(direct_model, X_test_raw)   # theo thứ tự lưu trong file
C_test_direct = np.empty_like(C_test_direct_raw)
C_test_direct[:, direct_order] = C_test_direct_raw               # sắp lại thành [LCC, SS, SC]
 
 
# --- Chỉ số sai số ---
def criteria_metrics(y_true, y_pred):
    rows_ = []
    for i, name in enumerate(CRITERIA_NAMES):
        t, p = y_true[:, i], y_pred[:, i]
        rmse = np.sqrt(mean_squared_error(t, p))
        rows_.append({
            "Criterion": name,
            "R2": r2_score(t, p),
            "MAE": mean_absolute_error(t, p),
            "RMSE": rmse,
            "NRMSE_mean_%": 100 * rmse / abs(t.mean()) if t.mean() != 0 else np.nan,
            "NRMSE_range_%": 100 * rmse / (t.max() - t.min()) if t.max() > t.min() else np.nan,
            "MaxErr": np.max(np.abs(p - t)),
        })
    return pd.DataFrame(rows_).set_index("Criterion")
 
 
m_indirect = criteria_metrics(C_test_true, C_test_indirect)
m_direct = criteria_metrics(C_test_true, C_test_direct)
 
crit_comparison = pd.concat(
    {"A: gián tiếp (4 output -> tiêu chí)": m_indirect,
     "B: trực tiếp (huấn luyện thẳng)": m_direct},
    axis=1,
)
print("\n" + "=" * 80)
print("SO SÁNH SAI SỐ TIÊU CHÍ TRÊN TẬP TEST (KRIGING)")
print("=" * 80)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(crit_comparison.to_string(float_format=lambda v: f"{v:.6g}"))
crit_comparison.to_csv("criteria_error_comparison_kriging.csv")
 
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
    ax.set_ylabel("Kriging prediction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("Done criteria comparison")