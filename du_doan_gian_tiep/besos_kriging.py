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
# BƯỚC 4: HUẤN LUYỆN KRIGING (1 GP CHO MỖI OUTPUT)
# ==========================================================
n_in = normed_train_data.shape[1]


def make_gp():
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_in),          # ARD: mỗi biến một length scale
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(1e-5, (1e-10, 1e-1))       # nugget
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS,
        random_state=SEED,
    )


import time
t1 = time.time()

gps = []
for j, name in enumerate(OUTPUT_NAMES):
    gp = make_gp().fit(normed_train_data, train_labels[:, j])
    gps.append(gp)
    print(f"\n{name}\n  kernel: {gp.kernel_}")
    print(f"  log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.3f}")
    print(f"\nThời gian bước này: {(time.time() - t1) / 60:.2f} phút")

print(f"\nThời gian huấn luyện Kriging: {(time.time() - t1) / 60:.2f} phút")
joblib.dump(
    {
        "gps": gps,
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "use_log_input": USE_LOG_INPUT,
    },
    "best_kriging_surrogate.joblib",
)
print("Done step 5")


def predict_norm(X_norm, return_std=False):
    X_norm = np.atleast_2d(X_norm)
    if not return_std:
        return np.column_stack([gp.predict(X_norm) for gp in gps])
    means, stds = [], []
    for gp in gps:
        m, s = gp.predict(X_norm, return_std=True)
        means.append(m)
        stds.append(s)
    return np.column_stack(means), np.column_stack(stds)


def Prediction(population_PV):
    return unnorm_output(predict_norm(Standardization(population_PV)))


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


# 5a. Cross-validation 5-fold trên tập train (thay cho validation_split)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_pred_norm = np.column_stack(
    [
        cross_val_predict(make_gp(), normed_train_data, train_labels[:, j], cv=cv)
        for j in range(len(OUTPUT_NAMES))
    ]
)
report_metrics("5-FOLD CROSS-VALIDATION ON TRAINING SET", Y_train, unnorm_output(cv_pred_norm))

# 5b. Trên chính tập train (Kriging nội suy nên R2 thường ~1)
y_train_pred = unnorm_output(predict_norm(normed_train_data))
report_metrics("PERFORMANCE ON TRAINING SET", Y_train, y_train_pred)
physical_check("PHYSICAL CONSTRAINT CHECK - TRAIN", y_train_pred)

# 5c. Tập test
y_test_pred_norm, y_test_std_norm = predict_norm(normed_test_data, return_std=True)
y_test_pred = unnorm_output(y_test_pred_norm)
summary_test = report_metrics("PERFORMANCE ON TEST SET", Y_test, y_test_pred)
physical_check("PHYSICAL CONSTRAINT CHECK - TEST", y_test_pred)
print("\n", summary_test)

# 5d. Độ bất định của Kriging trên tập test (đơn vị gốc)
print("\nĐộ lệch chuẩn dự đoán trung bình (đơn vị gốc):")
for i, name in enumerate(OUTPUT_NAMES):
    print(f"  {name}: {np.mean(y_test_std_norm[:, i] * output_std[i]):.6e}")

# 5e. Parity plot
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for i, ax in enumerate(axes.ravel()):
    ax.scatter(Y_test[:, i], y_test_pred[:, i], s=20, edgecolors="black", linewidths=0.3)
    lim = [min(Y_test[:, i].min(), y_test_pred[:, i].min()),
           max(Y_test[:, i].max(), y_test_pred[:, i].max())]
    ax.plot(lim, lim, "--", linewidth=1)
    ax.set_title(OUTPUT_NAMES[i])
    ax.set_xlabel("True")
    ax.set_ylabel("Kriging prediction")
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

    output_ANN = Prediction(population_PV)
    annual_grid_cost = output_ANN[:, 3]

    # Hiện giá tiền điện mua từ lưới trong 20 năm
    present_value_grid_cost = np.zeros(len(population_PV), dtype=float)
    for year in range(1, time + 1):
        present_value_grid_cost += annual_grid_cost / (1 + discount_rate) ** year

    LCC = initial_PV_cost + present_value_grid_cost
    SS = safe_div(output_ANN[:, 2], output_ANN[:, 0])
    SC = safe_div(output_ANN[:, 2], output_ANN[:, 1])
    return LCC, SS, SC, annual_grid_cost


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
        "Hãy kiểm tra chất lượng mô hình Kriging hoặc nới ràng buộc."
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
ax.set_title("Three-Dimensional Pareto-Optimal Solutions (Kriging)", fontsize=14, pad=18)
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
# BƯỚC 10: THIẾT KẾ ĐẶC TRƯNG + ĐỘ TIN CẬY CỦA KRIGING
# ==========================================================
representative_designs = {
    "Maximum SS": X_pareto[idx_max_ss],
    "Maximum SC": X_pareto[idx_max_sc],
    "Minimum LCC": X_pareto[idx_min_lcc],
    "Knee point": X_pareto[idx_knee],
}

print("\nBiến thiết kế của các nghiệm đặc trưng")
for name, x in representative_designs.items():
    mean_n, std_n = predict_norm(Standardization(x.reshape(1, -1)), return_std=True)
    rel_unc = (std_n[0] * output_std) / np.maximum(np.abs(unnorm_output(mean_n)[0]), 1.0)
    print(f"\n{name}:")
    print(x)
    print("Độ bất định tương đối của Kriging (Facility, Produced, E_self, Cost):")
    print(np.round(rel_unc * 100, 4), "%")

# Lưu để kiểm chứng lại bằng EnergyPlus
pd.DataFrame(X_pareto, columns=inputs.columns).assign(
    SS=SS_pareto, SC=SC_pareto, LCC=LCC_pareto
).to_csv("pareto_solutions_kriging.csv", index=False)
print("Done step 8")
