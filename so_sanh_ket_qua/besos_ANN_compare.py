import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd 
import numpy as np 

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

print("Done step 1")

# ==========================================================
# [PATCH] Cột output vật lý và cột label dùng để train.
#
# Mạng KHÔNG còn dự đoán E_self trực tiếp. Nó dự đoán tỉ lệ
#     r = E_self / min(Facility, Produced)   (luôn trong [0, 1])
# rồi E_self được dựng lại bằng to_physical(), nên
#     E_self <= Facility  và  E_self <= Produced
# luôn đúng => SS <= 1 và SC <= 1 theo cấu trúc.
# ==========================================================
PHYS_COLS = [
    "Electricity:Facility",
    "ElectricityProduced:Facility",
    "E_self",
    "Annual_grid_cost(VND)"
]
LABEL_COLS = [
    "Electricity:Facility",
    "ElectricityProduced:Facility",
    "E_self_ratio",
    "Annual_grid_cost(VND)"
]

def add_self_ratio(df):
    denom = np.minimum(
        df["Electricity:Facility"],
        df["ElectricityProduced:Facility"]
    )
    df = df.copy()
    df["E_self_ratio"] = np.clip(
        df["E_self"] / np.maximum(denom, 1.0),
        0.0,
        1.0
    )
    return df

# BIỂU GIÁ ĐIỆN THEO GIỜ
# Đơn vị: VNĐ/kWh
# ==========================================================

def get_electricity_rate(hour):

    # Giờ thấp điểm: 22:00 đến trước 04:00
    if (hour >= 22) or (hour < 4):
        return 1918

    # Giờ cao điểm
    elif ((hour >= 9.5) and (hour < 11.5)) or \
         ((hour >= 13.5) and (hour < 17)):
        return 3152

    # Giờ bình thường
    else:
        return 5422


# ==========================================================
# TÍNH OUTPUT TỔNG NĂM
# ==========================================================

annual_results_test = []

for _, row in outputs_hourly_test.iterrows():

    # Công suất phụ tải từng giờ, đơn vị W
    load = np.asarray(
        row["Electricity:Facility"],
        dtype=float
    )

    # Công suất PV sản xuất từng giờ, đơn vị W
    pv = np.asarray(
        row["ElectricityProduced:Facility"],
        dtype=float
    )

    # Kiểm tra hai chuỗi có cùng độ dài không
    if len(load) != len(pv):
        raise ValueError(
            "Electricity:Facility và "
            "ElectricityProduced:Facility không cùng độ dài."
        )

    # ------------------------------------------------------
    # 1. Điện PV tự tiêu thụ tại từng giờ
    # ------------------------------------------------------

    e_self_hourly = np.minimum(load, pv)

    # ------------------------------------------------------
    # 2. Công suất mua từ lưới tại từng giờ, đơn vị W
    # ------------------------------------------------------

    grid_power_hourly = np.maximum(
        load - pv,
        0
    )

    # Tương đương với:
    #
    # grid_power_hourly = np.maximum(load - pv, 0)

    # ------------------------------------------------------
    # 3. Đổi công suất W thành điện năng kWh
    #
    # Bước thời gian = 1 giờ
    # Wh = W × 1 giờ
    # kWh = Wh / 1000
    # ------------------------------------------------------

    grid_energy_hourly_kWh = (
        grid_power_hourly / (1000*3600)
    )

    # ------------------------------------------------------
    # 4. Xác định giờ trong ngày
    # ------------------------------------------------------

    hours = np.arange(len(load)) % 24

    # ------------------------------------------------------
    # 5. Giá điện của từng giờ
    # ------------------------------------------------------

    electricity_rate_hourly = np.array(
        [
            get_electricity_rate(hour)
            for hour in hours
        ],
        dtype=float
    )

    # ------------------------------------------------------
    # 6. Tiền mua điện lưới tại từng giờ
    # Đơn vị: VNĐ
    # ------------------------------------------------------

    grid_cost_hourly = (
        grid_energy_hourly_kWh
        * electricity_rate_hourly
    )

    # ------------------------------------------------------
    # 7. Tổng tiền điện mua từ lưới trong một năm
    # ------------------------------------------------------

    annual_grid_cost = grid_cost_hourly.sum()

    # ------------------------------------------------------
    # 8. Các output tổng năm
    # ------------------------------------------------------

    annual_results_test.append([
        load.sum(),
        pv.sum(),
        e_self_hourly.sum(),
        annual_grid_cost
    ])


annual_outputs_test = pd.DataFrame(
    annual_results_test,
    columns=[
        "Electricity:Facility",
        "ElectricityProduced:Facility",
        "E_self",
        "Annual_grid_cost(VND)"
    ]
)

annual_outputs_test = add_self_ratio(annual_outputs_test)  # [PATCH]
print(annual_outputs_test)
print("Done step 2")

inputs_test = inputs_all_test.iloc[:,32:40]
print(inputs_test)
print("Done step 3")

dataset_test = inputs_test.join(annual_outputs_test) ### Join ghép bảng input và output theo chiều ngang
dataset_test.head(20)
print("Done step 4")

# BIỂU GIÁ ĐIỆN THEO GIỜ
# Đơn vị: VNĐ/kWh
# ==========================================================

def get_electricity_rate(hour):

    # Giờ thấp điểm: 22:00 đến trước 04:00
    if (hour >= 22) or (hour < 4):
        return 1918

    # Giờ cao điểm
    elif ((hour >= 9.5) and (hour < 11.5)) or \
         ((hour >= 13.5) and (hour < 17)):
        return 3152

    # Giờ bình thường
    else:
        return 5422


# ==========================================================
# TÍNH OUTPUT TỔNG NĂM
# ==========================================================

annual_results = []

for _, row in outputs_hourly.iterrows():

    # Công suất phụ tải từng giờ, đơn vị W
    load = np.asarray(
        row["Electricity:Facility"],
        dtype=float
    )

    # Công suất PV sản xuất từng giờ, đơn vị W
    pv = np.asarray(
        row["ElectricityProduced:Facility"],
        dtype=float
    )

    # Kiểm tra hai chuỗi có cùng độ dài không
    if len(load) != len(pv):
        raise ValueError(
            "Electricity:Facility và "
            "ElectricityProduced:Facility không cùng độ dài."
        )

    # ------------------------------------------------------
    # 1. Điện PV tự tiêu thụ tại từng giờ
    # ------------------------------------------------------

    e_self_hourly = np.minimum(load, pv)

    # ------------------------------------------------------
    # 2. Công suất mua từ lưới tại từng giờ, đơn vị W
    # ------------------------------------------------------

    grid_power_hourly = np.maximum(
        load - pv,
        0
    )

    # Tương đương với:
    #
    # grid_power_hourly = np.maximum(load - pv, 0)

    # ------------------------------------------------------
    # 3. Đổi công suất W thành điện năng kWh
    #
    # Bước thời gian = 1 giờ
    # Wh = W × 1 giờ
    # kWh = Wh / 1000
    # ------------------------------------------------------

    grid_energy_hourly_kWh = (
        grid_power_hourly / (1000*3600)
    )

    # ------------------------------------------------------
    # 4. Xác định giờ trong ngày
    # ------------------------------------------------------

    hours = np.arange(len(load)) % 24

    # ------------------------------------------------------
    # 5. Giá điện của từng giờ
    # ------------------------------------------------------

    electricity_rate_hourly = np.array(
        [
            get_electricity_rate(hour)
            for hour in hours
        ],
        dtype=float
    )

    # ------------------------------------------------------
    # 6. Tiền mua điện lưới tại từng giờ
    # Đơn vị: VNĐ
    # ------------------------------------------------------

    grid_cost_hourly = (
        grid_energy_hourly_kWh
        * electricity_rate_hourly
    )

    # ------------------------------------------------------
    # 7. Tổng tiền điện mua từ lưới trong một năm
    # ------------------------------------------------------

    annual_grid_cost = grid_cost_hourly.sum()

    # ------------------------------------------------------
    # 8. Các output tổng năm
    # ------------------------------------------------------

    annual_results.append([
        load.sum(),
        pv.sum(),
        e_self_hourly.sum(),
        annual_grid_cost
    ])


annual_outputs = pd.DataFrame(
    annual_results,
    columns=[
        "Electricity:Facility",
        "ElectricityProduced:Facility",
        "E_self",
        "Annual_grid_cost(VND)"
    ]
)

annual_outputs = add_self_ratio(annual_outputs)  # [PATCH]
print(annual_outputs)

print("Done step 5")

inputs = inputs_all.iloc[:,32:40]
print(inputs)

print("Done step 6")

dataset = inputs.join(annual_outputs) ### Join ghép bảng input và output theo chiều ngang
print(dataset.head(20))

print("Done step 7")

train_stats = dataset[inputs.columns]
train_stats = train_stats.describe() ### Tạo bảng thống kê mô tả gồm tứ phân vị và count 
train_stats = train_stats.transpose() ### Chuyển vị dataframe
print(train_stats)

print("Done step 8")

# use the stats we calculated to do the normalization on the input.
def norm_input(x):
    return (x - train_stats['mean']) / train_stats['std'] ### Chuẩn hoá dữ liệu đưa dữ liệu về cùng 1 thang đo nhỏ hơn 

def unnorm_input(x):
    return (x* train_stats['std'])+ train_stats['mean'] ### Khôi phục dữ liệu gốc sau khi chuẩn hoá 
normed_train_data = norm_input(dataset[inputs.columns])
normed_test_data = norm_input(dataset_test[inputs_test.columns])

training_labels = dataset[LABEL_COLS]
training_labels = training_labels.describe()
training_labels = training_labels.transpose()
training_labels

print("Done step 9")

def norm_output(x):
    return (x - training_labels["mean"]) /training_labels["std"] ### Chuẩn hoá output của dữ liệu train 

def unnorm_output(x):
    x = np.asarray(x, dtype=float)
    return (x*training_labels["std"].to_numpy(dtype=float))+  training_labels["mean"].to_numpy(dtype=float)
train_labels = norm_output(dataset[LABEL_COLS])
test_labels  = norm_output(dataset_test[LABEL_COLS])

print("Done step 10")

import tensorflow as tf

output_mean = tf.constant(
    training_labels["mean"].to_numpy(dtype=np.float32),
    dtype=tf.float32)

output_std = tf.constant(
    training_labels["std"].to_numpy(dtype=np.float32),
    dtype=tf.float32)

print("Done step 11")
    
    ### Hàm truy Hồi dữ liệu đầu ra cho mô hình ANN 
def retrieve(output_ANN_norm):
    output_ANN = (output_ANN_norm * output_std+ output_mean)
    return output_ANN
    
### [PATCH] Đưa output thô của mạng về đại lượng vật lý hợp lệ
def to_physical(out):
    """
    out: (n, 4) = [Facility, Produced, E_self_ratio, Annual_grid_cost]
         đã giải chuẩn hóa.
    Trả về (n, 4) = [Facility, Produced, E_self, Annual_grid_cost]
    với E_self <= min(Facility, Produced) được đảm bảo.
    """
    out = np.asarray(out, dtype=float)
    facility = np.maximum(out[:, 0], 0.0)
    produced = np.maximum(out[:, 1], 0.0)
    ratio = np.clip(out[:, 2], 0.0, 1.0)
    e_self = ratio * np.minimum(facility, produced)
    cost = np.maximum(out[:, 3], 0.0)
    return np.column_stack([facility, produced, e_self, cost])

print("Done step 12")

# ==========================================================
# SO SÁNH SAI SỐ CỦA 3 TIÊU CHÍ (LCC, SS, SC) - ANN, DÙNG MÔ HÌNH ĐÃ LƯU
#   A) Gián tiếp : ANN 4 output (Facility, Produced, E_self_ratio, Cost)
#                  -> to_physical -> tính LCC/SS/SC từ dự đoán
#   B) Trực tiếp : ANN huấn luyện thẳng cho [LCC, SS, SC]
# Cả hai được so với tiêu chí tính từ dữ liệu test thật.
#
# DÁN KHỐI NÀY ngay sau  print("Done step 12")  (sau hàm to_physical, TRƯỚC phần
# định nghĩa mạng / keras-tuner). Khối này KHÔNG huấn luyện lại: nó nạp hai file
# .keras đã lưu. Nếu chỉ cần phần so sánh, chạy một bản sao của chương trình
# dừng ngay sau khối này (bỏ qua tuner.search và model.fit).
#
# Cần có sẵn: dataset, dataset_test, inputs, inputs_test, train_stats,
# unnorm_output, to_physical, PHYS_COLS, tf.
#
# Lưu ý: file .keras KHÔNG chứa thống kê chuẩn hóa. Khối này giả định:
#   - input của cả hai mạng được chuẩn hóa bằng train_stats (mean/std của train hiện tại);
#   - mạng trực tiếp chuẩn hóa output bằng mean/std (ddof=1, như describe())
#     của [LCC, SS, SC] tính trên tập train hiện tại.
# Hai mạng phải được huấn luyện trên ĐÚNG tập train này (cùng SPLIT_SEED,
# TEST_FRACTION), nếu không điểm test có thể đã nằm trong dữ liệu huấn luyện.
# ==========================================================
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
INDIRECT_MODEL_PATH = "D:/da2/4_best_ann_surrogate.keras"            # 4 output
DIRECT_MODEL_PATH = "D:/da2/3_best_ann_surrogate.keras"       # <-- đổi: 3 tiêu chí
CRITERIA_NAMES = ["LCC", "SS", "SC"]
MIN_TRAIN_R2 = 0.90   # ngưỡng kiểm tra: mô hình nạp vào phải khớp dữ liệu train này
STRICT_CHECK = True   # True: dừng nếu không đạt ngưỡng; False: chỉ cảnh báo
 
 
def safe_div(a, b):
    return a / np.maximum(b, 1.0)
 
 
def criteria_from_outputs(X, Y, years=20, discount_rate=0.1):
    """Cùng công thức với calculate_LCC_SS_SC nhưng nhận sẵn 4 output vật lý Y
    [Facility, Produced, E_self, Cost] (thật hoặc dự đoán). Trả về (n, 3): LCC, SS, SC."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
 
    initial_PV_cost = (X / 198).sum(axis=1) * 2_500_000 * 1.15
    annuity = sum(1.0 / (1 + discount_rate) ** t for t in range(1, years + 1))
    LCC = initial_PV_cost + Y[:, 3] * annuity
    SS = safe_div(Y[:, 2], Y[:, 0])
    SC = safe_div(Y[:, 2], Y[:, 1])
    return np.column_stack([LCC, SS, SC])
 
 
def fail_or_warn(msg):
    if STRICT_CHECK:
        raise AssertionError(msg)
    print("CẢNH BÁO:", msg)
 
 
# --- Dữ liệu thô (đơn vị gốc) và giá trị thật ---
X_train_raw = dataset[inputs.columns].to_numpy(dtype=float)
X_test_raw = dataset_test[inputs_test.columns].to_numpy(dtype=float)
Y_train_phys = dataset[PHYS_COLS].to_numpy(dtype=float)
Y_test_phys = dataset_test[PHYS_COLS].to_numpy(dtype=float)
 
C_train_true = criteria_from_outputs(X_train_raw, Y_train_phys)
C_test_true = criteria_from_outputs(X_test_raw, Y_test_phys)
 
# --- Chuẩn hóa input (giống hệt chương trình gốc) ---
x_mean = train_stats["mean"].to_numpy(dtype=float)
x_std = train_stats["std"].to_numpy(dtype=float)
Xn_train = (X_train_raw - x_mean) / x_std
Xn_test = (X_test_raw - x_mean) / x_std
 
# --- Nạp hai mạng đã lưu ---
indirect_net = tf.keras.models.load_model(INDIRECT_MODEL_PATH)
direct_net = tf.keras.models.load_model(DIRECT_MODEL_PATH)
assert indirect_net.output_shape[-1] == len(PHYS_COLS), \
    f"Mạng gián tiếp có {indirect_net.output_shape[-1]} output, cần {len(PHYS_COLS)}"
assert direct_net.output_shape[-1] == len(CRITERIA_NAMES), \
    f"Mạng trực tiếp có {direct_net.output_shape[-1]} output, cần {len(CRITERIA_NAMES)}"
 
# ==========================================================
# A) GIÁN TIẾP: 4 output -> tiêu chí
# ==========================================================
def predict_physical(Xn):
    raw = indirect_net.predict(Xn, verbose=0)
    return to_physical(unnorm_output(raw))   # đúng pipeline của chương trình gốc
 
 
Y_train_pred_phys = predict_physical(Xn_train)
r2_ind = [r2_score(Y_train_phys[:, i], Y_train_pred_phys[:, i]) for i in range(len(PHYS_COLS))]
print("[Indirect] R2 trên tập train (4 output):", np.round(r2_ind, 4))
if min(r2_ind) < MIN_TRAIN_R2:
    fail_or_warn(f"[Indirect] R2 train thấp nhất = {min(r2_ind):.3f} < {MIN_TRAIN_R2}: "
                 "file .keras không khớp dữ liệu/chuẩn hóa hiện tại")
 
C_test_indirect = criteria_from_outputs(X_test_raw, predict_physical(Xn_test))
 
# ==========================================================
# B) TRỰC TIẾP: mạng cho [LCC, SS, SC]
# Thứ tự cột output của mạng đã lưu có thể khác [LCC, SS, SC], nên tự dò:
# thử mọi hoán vị, giải chuẩn hóa bằng mean/std của tiêu chí tương ứng và chọn
# hoán vị cho R2 trên tập train cao nhất.
# ==========================================================
c_mean = C_train_true.mean(axis=0)
c_std = C_train_true.std(axis=0, ddof=1)
c_std[c_std == 0] = 1.0
 
direct_norm_train = direct_net.predict(Xn_train, verbose=0)
 
best_perm, best_score, best_r2 = None, -np.inf, None
for perm in itertools.permutations(range(len(CRITERIA_NAMES))):
    p = list(perm)                                   # cột k của mạng <-> tiêu chí p[k]
    pred = direct_norm_train * c_std[p] + c_mean[p]
    r2s = [r2_score(C_train_true[:, p[k]], pred[:, k]) for k in range(len(p))]
    if np.mean(r2s) > best_score:
        best_perm, best_score, best_r2 = p, np.mean(r2s), r2s
 
print("[Direct] thứ tự output trong file:", [CRITERIA_NAMES[k] for k in best_perm],
      "-> sẽ được sắp lại thành", CRITERIA_NAMES)
print("[Direct] R2 trên tập train theo từng tiêu chí (theo thứ tự cột của mạng):",
      np.round(best_r2, 4))
if min(best_r2) < MIN_TRAIN_R2:
    fail_or_warn(f"[Direct] R2 train thấp nhất = {min(best_r2):.3f} < {MIN_TRAIN_R2} với mọi thứ tự. "
                 "Nhiều khả năng mạng trực tiếp dùng cách chuẩn hóa output khác "
                 "(không phải mean/std của [LCC, SS, SC] trên train hiện tại) hoặc định nghĩa "
                 "tiêu chí khác.")
 
direct_norm_test = direct_net.predict(Xn_test, verbose=0)
C_test_direct = np.empty_like(direct_norm_test)
C_test_direct[:, best_perm] = direct_norm_test * c_std[best_perm] + c_mean[best_perm]
 
 
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
print("SO SÁNH SAI SỐ TIÊU CHÍ TRÊN TẬP TEST (ANN)")
print("=" * 80)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(crit_comparison.to_string(float_format=lambda v: f"{v:.6g}"))
crit_comparison.to_csv("criteria_error_comparison_ann.csv")
 
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
    ax.set_xlabel("True (từ dữ liệu test)")
    ax.set_ylabel("ANN prediction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("Done criteria comparison")
