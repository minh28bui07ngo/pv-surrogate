import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd 
import numpy as np 

INPUT_PKL  = "D:/da2/5000/inputs5000.pkl"    # <-- đổi thành tên file input tổng của bạn
OUTPUT_PKL = "D:/da2/5000/outputs5000_new.pkl"   # <-- file output MỚI (3 cột SS, SC, LCC) tạo bởi convert_outputs.py

TEST_FRACTION = 0.20          # 20% test, 80% train
SPLIT_SEED    = 42            # cố định seed để mỗi lần chạy chia giống nhau
STRATIFY_BY_TOTAL_PV = True   # chia đều theo tổng công suất PV (xem split_indices)

inputs_full  = pd.read_pickle(INPUT_PKL)
outputs_full = pd.read_pickle(OUTPUT_PKL)

# File output mới phải có đúng 3 cột SS, SC, LCC
assert list(outputs_full.columns) == ["SS", "SC", "LCC"], (
    f"File output phải có các cột ['SS', 'SC', 'LCC'], nhưng đang là {list(outputs_full.columns)}. "
    "Hãy chạy convert_outputs.py để tạo file output mới."
)

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
outputs_train_df      = outputs_full.iloc[train_idx].reset_index(drop=True)
inputs_all_test     = inputs_full.iloc[test_idx].reset_index(drop=True)
outputs_test_df = outputs_full.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(inputs_all)} mẫu | Test: {len(inputs_all_test)} mẫu "
      f"({len(inputs_all_test)/len(inputs_full):.0%})")

print("Done step 1")

# ==========================================================
# 3 đại lượng cần dự đoán (đã tính sẵn bằng convert_outputs.py):
#     SS  = E_self / Electricity:Facility          (luôn trong [0, 1])
#     SC  = E_self / ElectricityProduced:Facility  (luôn trong [0, 1])
#     LCC = chi phí đầu tư PV ban đầu + giá trị hiện tại của tiền điện mua lưới 20 năm
# Mạng dự đoán trực tiếp 3 đại lượng này. to_physical() ép SS, SC về [0, 1] và
# LCC >= chi phí đầu tư ban đầu (tương đương tiền điện mua lưới >= 0), nên SS <= 1,
# SC <= 1 luôn đúng theo cấu trúc.
# ==========================================================
PHYS_COLS  = ["SS", "SC", "LCC"]
LABEL_COLS = PHYS_COLS          # SS, SC đã là tỉ lệ nên không cần đổi nhãn như trước

# Hằng số chi phí đầu tư (cùng công thức với calculate_LCC_SS_SC cũ)
W_PER_PANEL_AREA = 198
COST_PER_AREA = 2_500_000
COST_FACTOR = 1.15

def initial_pv_cost(population_PV):
    """Chi phí đầu tư PV ban đầu (VND) = tổng công suất / 198 * 2.5e6 * 1.15."""
    population_PV = np.atleast_2d(np.asarray(population_PV, dtype=float))
    return population_PV.sum(axis=1) / W_PER_PANEL_AREA * COST_PER_AREA * COST_FACTOR

# Output đã được tính sẵn (SS, SC, LCC) nên không cần tính lại từ dữ liệu theo giờ.
annual_outputs_test = outputs_test_df
print(annual_outputs_test)
print("Done step 2")

inputs_test = inputs_all_test.iloc[:,32:40]
print(inputs_test)
print("Done step 3")

dataset_test = inputs_test.join(annual_outputs_test) ### Join ghép bảng input và output theo chiều ngang
dataset_test.head(20)
print("Done step 4")

annual_outputs = outputs_train_df
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
    
### Đưa output thô của mạng về đại lượng vật lý hợp lệ
def to_physical(out, pv_capacity):
    """
    out: (n, 3) = [SS, SC, LCC] đã giải chuẩn hóa.
    pv_capacity: (n, 8) công suất PV1..PV8 tương ứng (để tính chi phí đầu tư ban đầu).
    Trả về (n, 3) với SS, SC trong [0, 1] và LCC >= chi phí đầu tư ban đầu.
    """
    out = np.asarray(out, dtype=float)
    SS = np.clip(out[:, 0], 0.0, 1.0)
    SC = np.clip(out[:, 1], 0.0, 1.0)
    LCC = np.maximum(out[:, 2], initial_pv_cost(pv_capacity))
    return np.column_stack([SS, SC, LCC])

print("Done step 12")


import tensorflow as tf   ### Bộ máy tính toán và huấn luyện mạng notron 
import tensorflow.keras as keras ## Keras là API ( GIAO THỨC ) Cấp cao của thư viện tensorflow
from tensorflow.keras import layers ### Dùng để tạo các lớp mạng 

def building_surrogate_modelling(hp):
    
    model = keras.Sequential() ## Khởi tạo mô hình mạng Notron kiểu tuần tự , đầu ra của lớp này là đầu vào của lớp sau 
    model.add(
        layers.Input(shape=(dataset[inputs.columns].shape[1],))
    )
    n_layers = hp.Int(
        "n_lớp_ẩn",
        min_value=1,
        max_value=4
    ) ## Số lớp ẩn có thể thử là từ 1 đến 4 lớp 
    for i in range(n_layers):
          units = hp.Int(
            f"units_{i}",
            min_value=16,
            max_value=128,
            step=16
        )

          activation = hp.Choice(
            f"activation_{i}",
            values=["relu", "tanh", "swish"]
        )
          l2_value = hp.Float(
          f"l2_{i}",
          min_value=1e-6,
          max_value=1e-3,
          sampling="log")

          l2_regularizer = keras.regularizers.l2(l2_value)

          model.add(
          layers.Dense(
          units=units,
          activation=activation,
          kernel_regularizer=l2_regularizer)) 

          use_batch_norm = hp.Boolean(
            f"batch_norm_{i}"
        )

          if use_batch_norm:
            model.add(layers.BatchNormalization())
              
          dropout_rate = hp.Float(
            f"dropout_{i}",
            min_value=0.0,
            max_value=0.20,
            step=0.05
        )

          if dropout_rate > 0:
            model.add(
                layers.Dropout(dropout_rate))
    model.add(
        layers.Dense(
            units=len(LABEL_COLS),
            activation="linear"
        )) 
    learning_rate = hp.Float(
        "learning_rate",
        min_value=1e-5,
        max_value=1e-2,
        sampling="log"
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )
    
    # [PATCH] Ràng buộc đã nằm trong cấu trúc output => chỉ cần MSE
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            keras.metrics.MeanSquaredError(
                name="mse"
            ),
            keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ]
    )

    return model
    
print("Done step 13")

import shutil
import os

# Xóa kết quả tuner cũ (nếu có)
if os.path.exists("surrogate_tuner"):
    shutil.rmtree("surrogate_tuner")
    
import keras_tuner as kt
tuner = kt.BayesianOptimization(
    hypermodel= building_surrogate_modelling,
    objective="val_loss",
    max_trials=40,
    executions_per_trial=2,
    overwrite=True,
    directory="surrogate_tuner",
    project_name="energyplus_ann"
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

print("Done step 14")

tuner.search(
    normed_train_data,
    train_labels,
    validation_split=0.15,
    epochs=500,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("Done step 15")

### Lưu lại mô hình chứa bộ siêu tham số tốt nhất 
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save("best_ann_surrogate.keras")
#from tensorflow import keras

#model_x = keras.models.load_model("best_ann_surrogate.keras")

print("Done step 16")

import time
t1 = time.time()

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0] ### Lấy bộ siêu tham số tốt nhất 
print(best_hp.get("n_lớp_ẩn"))
print(best_hp.get("units_0"))
print(best_hp.get("activation_0"))
print(best_hp.get("learning_rate"))
#### Thực hiện train lại mô hình với cấu trúc mạng thiết kế từ bộ siêu tham số tốt nhất đối với mẫu dữ liệu 
model = building_surrogate_modelling(best_hp)
### Train lại mô hình
history = model.fit(
    normed_train_data,
    train_labels,
    validation_split=0.15,
    epochs=1000,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

t2 = time.time()
print(f"Thời gian mô phỏng: {(t2 - t1) / 60:.2f} phút")

print("Done step 17")

import matplotlib.pyplot as plt
import numpy as np

# Trục epoch
epochs = np.arange(1, len(history.history["loss"]) + 1)

# ==========================================================
# 1. LOSS: TRAINING VÀ VALIDATION
# ==========================================================
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history.history["val_loss"],
    label="Validation Loss"
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
best_val_loss = np.min(history.history["val_loss"])

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Best epoch:", best_epoch)
print("Minimum validation loss:", best_val_loss)


# ==========================================================
# 2. MSE: TRAINING VÀ VALIDATION
# ==========================================================
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    history.history["mse"],
    label="Training MSE"
)

plt.plot(
    epochs,
    history.history["val_mse"],
    label="Validation MSE"
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation MSE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ==========================================================
# 3. MAE: TRAINING VÀ VALIDATION
# ==========================================================
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    history.history["mae"],
    label="Training MAE"
)

plt.plot(
    epochs,
    history.history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training and Validation MAE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Done step 18")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================================
# 1. TÁCH INPUT VÀ OUTPUT TỪ TOÀN BỘ DATASET
# ==========================================================

X_dataset = dataset[inputs.columns]

y_dataset = dataset[PHYS_COLS]


# ==========================================================
# 2. CHUẨN HÓA INPUT BẰNG THỐNG KÊ ĐÃ DÙNG KHI TRAIN
# ==========================================================

X_dataset_norm = (
    X_dataset.to_numpy(dtype=float)
    - train_stats["mean"].to_numpy(dtype=float)
) / train_stats["std"].to_numpy(dtype=float)


# ==========================================================
# 3. DỰ ĐOÁN TRÊN TOÀN BỘ DATASET
# ==========================================================

y_dataset_pred_norm = best_model.predict(
    X_dataset_norm,
    verbose=0
)

# Giải chuẩn hóa dự đoán
y_dataset_pred_raw = unnorm_output(
    y_dataset_pred_norm
)

# Dự đoán thô ngoài miền vật lý = mạng "muốn" vi phạm bao nhiêu mẫu
ss_raw, sc_raw = y_dataset_pred_raw[:, 0], y_dataset_pred_raw[:, 1]
print(
    "Raw SS ngoài [0,1] (trước khi ép):",
    int(np.sum((ss_raw < 0) | (ss_raw > 1))), "/", len(ss_raw)
)
print(
    "Raw SC ngoài [0,1] (trước khi ép):",
    int(np.sum((sc_raw < 0) | (sc_raw > 1))), "/", len(sc_raw)
)
print(
    "Raw LCC < chi phí đầu tư ban đầu (trước khi ép):",
    int(np.sum(y_dataset_pred_raw[:, 2] < initial_pv_cost(X_dataset.to_numpy()))),
    "/", len(ss_raw)
)

# Đưa về đại lượng vật lý hợp lệ
y_dataset_pred = to_physical(y_dataset_pred_raw, X_dataset.to_numpy(dtype=float))

# Giá trị thật ở đơn vị gốc
y_dataset_true = y_dataset.to_numpy(dtype=float)


# ==========================================================
# 4. KIỂM TRA SHAPE VÀ THỨ TỰ OUTPUT
# ==========================================================

print("Shape input:", X_dataset_norm.shape)
print("Shape true :", y_dataset_true.shape)
print("Shape pred :", y_dataset_pred.shape)

print("\nThứ tự output:")
print(PHYS_COLS)


# ==========================================================
# 5. TÍNH R², MAE, RMSE CHO TỪNG OUTPUT
# ==========================================================

results_dataset = []

print("\n" + "=" * 80)
print("BEST MODEL PERFORMANCE ON FULL DEVELOPMENT DATASET")
print("=" * 80)

for i, name in enumerate(PHYS_COLS):

    true = y_dataset_true[:, i]
    pred = y_dataset_pred[:, i]

    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))

    true_mean = np.mean(true)
    true_std = np.std(true, ddof=1)
    true_range = np.max(true) - np.min(true)

    nrmse_mean = (
        rmse / abs(true_mean)
        if abs(true_mean) > 0
        else np.nan
    )

    nrmse_std = (
        rmse / true_std
        if true_std > 0
        else np.nan
    )

    nrmse_range = (
        rmse / true_range
        if true_range > 0
        else np.nan
    )

    correlation = np.corrcoef(true, pred)[0, 1]

    results_dataset.append({
        "Output": name,
        "R²": r2,
        "Pearson_r": correlation,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE_mean_percent": nrmse_mean * 100,
        "RMSE_over_std": nrmse_std,
        "NRMSE_range_percent": nrmse_range * 100
    })

    print(f"\n{name}")
    print(f"R²                 : {r2:.6f}")
    print(f"Pearson correlation: {correlation:.6f}")
    print(f"MAE                : {mae:.6e}")
    print(f"RMSE               : {rmse:.6e}")
    print(f"NRMSE(mean)        : {nrmse_mean:.6%}")
    print(f"RMSE/std           : {nrmse_std:.6f}")
    print(f"NRMSE(range)       : {nrmse_range:.6%}")


results_dataset_df = pd.DataFrame(
    results_dataset
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(results_dataset_df)


# ==========================================================
# 6. KIỂM TRA RIÊNG TỪNG OUTPUT
# ==========================================================

print("\n" + "=" * 80)
print("OUTPUT DIAGNOSTIC (FULL DATASET)")
print("=" * 80)

for k, name in enumerate(PHYS_COLS):
    t_k = y_dataset_true[:, k]
    p_k = y_dataset_pred[:, k]
    e_k = p_k - t_k
    print(f"\n{name}")
    print("True mean :", t_k.mean())
    print("Pred mean :", p_k.mean())
    print("True std  :", t_k.std(ddof=1))
    print("Pred std  :", p_k.std(ddof=1))
    print("Mean error:", e_k.mean())
    print("Std error :", e_k.std(ddof=1))
    print("True-Pred correlation:", np.corrcoef(t_k, p_k)[0, 1])


# ==========================================================
# 7. KIỂM TRA RÀNG BUỘC VẬT LÝ TRÊN DATASET
# ==========================================================

SS_pred_dataset = y_dataset_pred[:, 0]
SC_pred_dataset = y_dataset_pred[:, 1]
LCC_pred_dataset = y_dataset_pred[:, 2]

finite_mask = (
    np.isfinite(SS_pred_dataset)
    & np.isfinite(SC_pred_dataset)
    & np.isfinite(LCC_pred_dataset)
)

ss_violation = finite_mask & (SS_pred_dataset > 1.0)
sc_violation = finite_mask & (SC_pred_dataset > 1.0)
lcc_violation = finite_mask & (
    LCC_pred_dataset < initial_pv_cost(X_dataset.to_numpy(dtype=float)) - 1e-6
)
any_violation = ss_violation | sc_violation | lcc_violation

print("\n" + "=" * 80)
print("PHYSICAL CONSTRAINT CHECK ON FULL DATASET")
print("=" * 80)

print(f"Tổng số mẫu: {len(y_dataset_pred)}")
print(f"SS > 1: {ss_violation.sum()} ({ss_violation.mean() * 100:.4f}%)")
print(f"SC > 1: {sc_violation.sum()} ({sc_violation.mean() * 100:.4f}%)")
print(f"LCC < chi phí đầu tư ban đầu: {lcc_violation.sum()} ({lcc_violation.mean() * 100:.4f}%)")
print(f"Vi phạm ít nhất một điều kiện: {any_violation.sum()} ({any_violation.mean() * 100:.4f}%)")
print(f"Giá trị không hữu hạn: {(~finite_mask).sum()}")
print(f"SS dự đoán : {SS_pred_dataset[finite_mask].min():.6f} → {SS_pred_dataset[finite_mask].max():.6f}")
print(f"SC dự đoán : {SC_pred_dataset[finite_mask].min():.6f} → {SC_pred_dataset[finite_mask].max():.6f}")
print(f"LCC dự đoán: {LCC_pred_dataset[finite_mask].min():,.2f} → {LCC_pred_dataset[finite_mask].max():,.2f}")

print("Done step 19")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# 1. DỰ ĐOÁN TRÊN TẬP TEST
# =====================================================

# Giá trị dự đoán (chuẩn hóa)
y_pred_norm = best_model.predict(normed_test_data, verbose=0)

# Giải chuẩn hóa
y_pred_raw = unnorm_output(y_pred_norm)
y_pred = to_physical(y_pred_raw, dataset_test[inputs_test.columns].to_numpy(dtype=float))

# Giá trị thật
y_true = dataset_test[PHYS_COLS].to_numpy(dtype=float)

# =====================================================
# 2. ĐÁNH GIÁ TỪNG OUTPUT
# =====================================================

output_names = list(PHYS_COLS)

print("="*80)
print("MODEL PERFORMANCE")
print("="*80)

for i, name in enumerate(output_names):

    true = y_true[:, i]
    pred = y_pred[:, i]

    r2 = r2_score(true, pred)

    mae = mean_absolute_error(true, pred)

    rmse = np.sqrt(mean_squared_error(true, pred))

    nrmse_mean = rmse / np.mean(true)

    nrmse_range = rmse / (np.max(true) - np.min(true))

    max_error = np.max(np.abs(pred - true))

    print(f"\n{name}")

    print(f"R²           : {r2:.6f}")
    print(f"MAE          : {mae:.6e}")
    print(f"RMSE         : {rmse:.6e}")
    print(f"NRMSE(mean)  : {nrmse_mean:.6%}")
    print(f"NRMSE(range) : {nrmse_range:.6%}")
    print(f"Max Error    : {max_error:.6e}")

# =====================================================
# 3. KIỂM TRA RÀNG BUỘC VẬT LÝ
# =====================================================

SS = y_pred[:,0]
SC = y_pred[:,1]
LCC = y_pred[:,2]

ss_violation = SS > 1
sc_violation = SC > 1

both_violation = ss_violation | sc_violation

print("\n")
print("="*80)
print("PHYSICAL CONSTRAINT CHECK")
print("="*80)

print(f"Total test samples : {len(y_pred)}")

print(f"\nSS > 1")
print(f"Number : {np.sum(ss_violation)}")
print(f"Ratio  : {100*np.mean(ss_violation):.4f}%")

print(f"\nSC > 1")
print(f"Number : {np.sum(sc_violation)}")
print(f"Ratio  : {100*np.mean(sc_violation):.4f}%")

print(f"\nSS > 1 OR SC > 1")
print(f"Number : {np.sum(both_violation)}")
print(f"Ratio  : {100*np.mean(both_violation):.4f}%")

# =====================================================
# 4. THỐNG KÊ SS, SC
# =====================================================

print("\n")
print("="*80)
print("PREDICTED SS / SC")
print("="*80)

print(f"SS : {SS.min():.6f}  -->  {SS.max():.6f}")
print(f"SC : {SC.min():.6f}  -->  {SC.max():.6f}")
print(f"LCC: {LCC.min():,.2f}  -->  {LCC.max():,.2f}")
lcc_violation = LCC < initial_pv_cost(dataset_test[inputs_test.columns].to_numpy(dtype=float)) - 1e-6
print(f"LCC < chi phí đầu tư ban đầu: {np.sum(lcc_violation)}")

# =====================================================
# 5. DATAFRAME TỔNG HỢP
# =====================================================

summary = []

for i, name in enumerate(output_names):

    true = y_true[:,i]
    pred = y_pred[:,i]

    summary.append([
        name,
        r2_score(true,pred),
        mean_absolute_error(true,pred),
        np.sqrt(mean_squared_error(true,pred))
    ])

summary = pd.DataFrame(
    summary,
    columns=[
        "Output",
        "R²",
        "MAE",
        "RMSE"
    ]
)

print("\n")
print(summary)


print("Done step 20")

for k, name in enumerate(output_names):
    error = y_pred[:,k] - y_true[:,k]
    print(f"\n{name}")
    print("mean error:", error.mean())
    print("std error :", error.std())
    print("mean y_true:", y_true[:,k].mean())
    print("mean y_pred:", y_pred[:,k].mean())

print("Done step 21: mean,std of error, mean of y_true and y_pred")

# ==========================================================
# 21b. ĐỘ QUAN TRỌNG CỦA TỪNG VÙNG PV + SO SÁNH DỰ ĐOÁN VỚI GIÁ TRỊ THẬT
# (tương đương bước 6d và 6e của chương trình XGBoost)
# ==========================================================
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. Feature importance bằng permutation importance
#    ANN không có feature_importances_ như XGBoost, nên dùng phương pháp
#    hoán vị: xáo trộn ngẫu nhiên MỘT cột input, đo mức tăng MSE của từng
#    output (trên thang chuẩn hóa để 3 output so sánh được với nhau).
#    Cột nào làm sai số tăng càng nhiều thì càng quan trọng.
#    Kết quả được chuẩn hóa để tổng = 1 cho mỗi output, giống XGBoost.
# ----------------------------------------------------------

PERM_REPEATS = 30
rng_perm = np.random.default_rng(42)

X_perm = normed_test_data.to_numpy(dtype=float)
Y_perm = test_labels.to_numpy(dtype=float)   # output đã chuẩn hóa


def mse_per_output(X):
    pred_norm = best_model.predict(X, verbose=0)
    return ((pred_norm - Y_perm) ** 2).mean(axis=0)


base_mse = mse_per_output(X_perm)

perm_increase = np.zeros((X_perm.shape[1], Y_perm.shape[1]))

for j in range(X_perm.shape[1]):
    increase = np.zeros((PERM_REPEATS, Y_perm.shape[1]))
    for r in range(PERM_REPEATS):
        X_shuffled = X_perm.copy()
        X_shuffled[:, j] = rng_perm.permutation(X_shuffled[:, j])
        increase[r] = mse_per_output(X_shuffled) - base_mse
    perm_increase[j] = increase.mean(axis=0)

# Giá trị âm (do nhiễu) coi như không quan trọng
perm_increase = np.clip(perm_increase, 0, None)

col_sum = perm_increase.sum(axis=0)
col_sum[col_sum == 0] = 1.0

imp_ann = pd.DataFrame(
    perm_increase / col_sum,
    index=inputs.columns,
    columns=[f"{n} [ANN]" for n in output_names]
)

print("\nFeature importance (permutation, tập test):")
print(imp_ann.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
imp_ann.plot.bar(ax=ax)
ax.set_ylabel("Importance")
ax.set_title("Feature importance per output (ANN, permutation)")
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2. Parity plot: dự đoán so với giá trị thật trên tập test
# ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

for i, ax in enumerate(axes.ravel()):
    ax.scatter(
        y_true[:, i],
        y_pred[:, i],
        s=20,
        edgecolors="black",
        linewidths=0.3
    )

    lim = [
        min(y_true[:, i].min(), y_pred[:, i].min()),
        max(y_true[:, i].max(), y_pred[:, i].max())
    ]

    ax.plot(lim, lim, "--", linewidth=1)
    ax.set_title(f"{output_names[i]} [ANN]")
    ax.set_xlabel("True")
    ax.set_ylabel("Prediction")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

print("Done step 21b: feature importance and parity plot")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ==========================================================
# 22. KIỂM TRA CHI TIẾT TỪNG OUTPUT
# ==========================================================

print("=" * 80)
print("1. OUTPUT COLUMN ORDER")
print("=" * 80)

print("train_labels columns:", list(train_labels.columns))
print("test_labels columns :", list(test_labels.columns))
print("\nOutput normalization statistics:")
print(training_labels)

for k, name in enumerate(PHYS_COLS):
    true_norm = test_labels[name].to_numpy(dtype=float)
    pred_norm = y_pred_norm[:, k]
    true_o = y_true[:, k]
    pred_o = y_pred[:, k]
    err = pred_o - true_o

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    print("Normalized  true mean/std:", true_norm.mean(), true_norm.std(ddof=1))
    print("Normalized  pred mean/std:", pred_norm.mean(), pred_norm.std(ddof=1))
    print("Normalized  pred min/max :", pred_norm.min(), pred_norm.max())

    print("Original    true mean/std:", true_o.mean(), true_o.std(ddof=1))
    print("Original    pred mean/std:", pred_o.mean(), pred_o.std(ddof=1))
    print("Original    true min/max :", true_o.min(), true_o.max())
    print("Original    pred min/max :", pred_o.min(), pred_o.max())

    print("Error mean / std / min / max / max-abs:",
          err.mean(), err.std(ddof=1), err.min(), err.max(), np.max(np.abs(err)))

    print("Pearson correlation:", np.corrcoef(true_o, pred_o)[0, 1])
    print("R²                 :", r2_score(true_o, pred_o))

    print("\nTrue vs predicted (20 mẫu đầu):")
    print(pd.DataFrame({
        "True_norm": true_norm,
        "Pred_norm": pred_norm,
        "True_original": true_o,
        "Pred_original": pred_o,
        "Error_original": err
    }).head(20))

    dev_o = dataset[name].to_numpy(dtype=float)
    tst_o = dataset_test[name].to_numpy(dtype=float)
    shift = dev_o.mean() - tst_o.mean()
    print(f"\nDevelopment: mean {dev_o.mean():.6e}, std {dev_o.std(ddof=1):.6e}, "
          f"min {dev_o.min():.6e}, max {dev_o.max():.6e}")
    print(f"Test       : mean {tst_o.mean():.6e}, std {tst_o.std(ddof=1):.6e}, "
          f"min {tst_o.min():.6e}, max {tst_o.max():.6e}")
    print("Mean difference / Development std:", shift / dev_o.std(ddof=1))

print("Done step 22")

corr = dataset.corr(numeric_only=True)

for name in PHYS_COLS:
    print(f"\nTương quan giữa các PV và {name}:")
    print(
        corr.loc[inputs.columns, name]
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )

print("Done step 23")

### Hàm chuẩn hóa dữ liệu đầu vào cho mô hình ANN 
def Standardization(population_PV):
    def norm_input(x):
        return (x - train_stats['mean'].to_numpy(dtype=float)) / train_stats['std'].to_numpy(dtype=float) ### Chuẩn hoá dữ liệu đưa dữ liệu về cùng 1 thang đo nhỏ hơn 
    inputs_ANN_norm = norm_input(population_PV)
    return inputs_ANN_norm
    
### Hàm chuyển thẳng từ Population_PV ra 3 đại lượng SS, SC, LCC
def Prediction(population_PV):
    inputs_ANN_norm = Standardization(population_PV)
    ## Dự đoán output bằng mô hình ANN
    output_ANN_norm = best_model.predict(
        inputs_ANN_norm,
        verbose=0
    )
    ### Truy xuất về dữ liệu thực rồi ép về miền vật lý hợp lệ
    output_ANN = to_physical(unnorm_output(output_ANN_norm), population_PV)
    return output_ANN

### Hàm LCC, SS, SC: mạng dự đoán trực tiếp cả ba đại lượng
def calculate_LCC_SS_SC(population_PV):
    output_ANN = Prediction(population_PV)
    SS = output_ANN[:, 0]
    SC = output_ANN[:, 1]
    LCC = output_ANN[:, 2]
    return LCC , SS , SC

print("Done step 24")

import numpy as np

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

### Khai báo giới hạn của các tham số công suất định mức 
lower_bounds = np.array([
    198.000621,
    198.000621,
    197.999648,
    197.999648,
    990.000337,
    990.000337,
    198.001915,
    198.001915
])
upper_bounds = np.array([
    73656.000079,
    73656.000079,
    36827.998902,
    36827.998902,
    96227.999467,
    96227.999467,
    69695.998978,
    69695.998978
])

print("Done step 25")

class PVProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=8,
            n_obj=3,
            n_ieq_constr=0,
            xl=lower_bounds,
            xu=upper_bounds)
    def _evaluate( self,X,out, *args,**kwargs):
        LCC , SS , SC  = calculate_LCC_SS_SC(X)
        out["F"] = np.column_stack([ -SS , -SC , LCC])
        
problem = PVProblem()

### NSGA II 
algorithm = NSGA2(
    pop_size = 6400,
    eliminate_duplicates=True
)

### Điều kiện dừng , dừng sau bao nhiêu thế hệ 
termination = get_termination(
    "n_gen",
    500)
    
### Optimization
result = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True,
    save_history=True)
    
print("Done step 26")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# LẤY KẾT QUẢ PARETO TỪ NSGA-II
# =========================================================

X_pareto = np.asarray(result.X, dtype=float)
F_pareto = np.asarray(result.F, dtype=float)

# Khôi phục giá trị thật của các hàm mục tiêu
SS_pareto  = -F_pareto[:, 0]
SC_pareto  = -F_pareto[:, 1]
LCC_pareto =  F_pareto[:, 2]

print("Số nghiệm Pareto:", len(X_pareto))
print("Shape X:", X_pareto.shape)
print("Shape F:", F_pareto.shape)

print(f"SS : {SS_pareto.min():.6f} → {SS_pareto.max():.6f}")
print(f"SC : {SC_pareto.min():.6f} → {SC_pareto.max():.6f}")
print(f"LCC: {LCC_pareto.min():,.2f} → {LCC_pareto.max():,.2f}")

print("Done step 27")

import numpy as np

SS_pareto = np.asarray(SS_pareto).reshape(-1)
SC_pareto = np.asarray(SC_pareto).reshape(-1)

ss_invalid = SS_pareto > 1.0
sc_invalid = SC_pareto > 1.0
any_invalid = ss_invalid | sc_invalid

n_total = len(SC_pareto)

print("Tổng số nghiệm Pareto:", n_total)

print(
    "SS > 1:",
    np.sum(ss_invalid),
    f"({100*np.mean(ss_invalid):.2f}%)"
)

print(
    "SC > 1:",
    np.sum(sc_invalid),
    f"({100*np.mean(sc_invalid):.2f}%)"
)

print(
    "Vi phạm ít nhất một điều kiện:",
    np.sum(any_invalid),
    f"({100*np.mean(any_invalid):.2f}%)"
)

if np.any(sc_invalid):
    print(
        "SC trung bình của các nghiệm vi phạm:",
        SC_pareto[sc_invalid].mean()
    )

    print(
        "SC lớn nhất:",
        SC_pareto[sc_invalid].max()
    )

    print(
        "Mức vượt trung bình:",
        (SC_pareto[sc_invalid] - 1.0).mean()
    )
    
# Nghiệm Pareto nằm trên biên bị ép (raw SS/SC >= 1 hoặc raw LCC < chi phí đầu tư)
# là dấu hiệu NSGA-II đang khai thác sai số surrogate -> nên kiểm chứng bằng EnergyPlus.
raw_pareto = unnorm_output(
    best_model.predict(Standardization(X_pareto), verbose=0)
)
ss_pareto_raw = raw_pareto[:, 0]
sc_pareto_raw = raw_pareto[:, 1]
lcc_pareto_raw = raw_pareto[:, 2]
n_edge = int(np.sum((ss_pareto_raw >= 1.0) | (sc_pareto_raw >= 1.0)))
n_lcc_edge = int(np.sum(lcc_pareto_raw < initial_pv_cost(X_pareto)))
print(
    f"Nghiệm Pareto có raw SS hoặc SC >= 1 (bị ép về 1): "
    f"{n_edge}/{len(ss_pareto_raw)}"
)
print(
    f"Raw SS lớn nhất: {ss_pareto_raw.max():.4f}, "
    f"raw SC lớn nhất: {sc_pareto_raw.max():.4f}"
)
print(
    f"Nghiệm Pareto có raw LCC < chi phí đầu tư ban đầu (bị ép): "
    f"{n_lcc_edge}/{len(lcc_pareto_raw)}"
)

print("Done step 28")

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# LẤY KẾT QUẢ PARETO TỪ PYMOO
# ==========================================================

F = np.asarray(result.F)

# Trong bài toán của bạn:
# out["F"] = np.column_stack([-SS, -SC, LCC])

SS = -F[:, 0]
SC = -F[:, 1]
LCC = F[:, 2]

print("Số nghiệm Pareto:", len(F))
print(f"SS : {SS.min():.6f} → {SS.max():.6f}")
print(f"SC : {SC.min():.6f} → {SC.max():.6f}")
print(f"LCC: {LCC.min():,.2f} → {LCC.max():,.2f}")


# ==========================================================
# 1. VẼ MẶT PARETO 3D
# ==========================================================

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

pareto_3d = ax.scatter(
    SS,
    SC,
    LCC,
    c=LCC,
    s=45,
    edgecolors="black",
    linewidths=0.3
)

ax.set_title(
    "Three-Dimensional Pareto Front",
    fontsize=15,
    pad=18
)

ax.set_xlabel(
    "Self-Sufficiency (SS)",
    fontsize=11,
    labelpad=10
)

ax.set_ylabel(
    "Self-Consumption (SC)",
    fontsize=11,
    labelpad=10
)

ax.set_zlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11,
    labelpad=10
)

colorbar = fig.colorbar(
    pareto_3d,
    ax=ax,
    shrink=0.72,
    pad=0.10
)

colorbar.set_label(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.tight_layout()
plt.show()


# ==========================================================
# 2. HÌNH CHIẾU LCC – SS
#    MÀU BIỂU DIỄN SC
# ==========================================================

idx_lcc = np.argsort(LCC)

plt.figure(figsize=(8, 6))

projection_1 = plt.scatter(
    LCC,
    SS,
    c=SC,
    s=45,
    edgecolors="black",
    linewidths=0.3
)

plt.plot(
    LCC[idx_lcc],
    SS[idx_lcc],
    linewidth=1.2,
    alpha=0.7
)

plt.title(
    "Pareto Front Projection: LCC versus SS",
    fontsize=14
)

plt.xlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.ylabel(
    "Self-Sufficiency (SS)",
    fontsize=11
)

colorbar = plt.colorbar(projection_1)

colorbar.set_label(
    "Self-Consumption (SC)",
    fontsize=11
)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ==========================================================
# 3. HÌNH CHIẾU LCC – SC
#    MÀU BIỂU DIỄN SS
# ==========================================================

plt.figure(figsize=(8, 6))

projection_2 = plt.scatter(
    LCC,
    SC,
    c=SS,
    s=45,
    edgecolors="black",
    linewidths=0.3
)

plt.plot(
    LCC[idx_lcc],
    SC[idx_lcc],
    linewidth=1.2,
    alpha=0.7
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.2,
    label="Physical limit: SC = 1"
)

plt.title(
    "Pareto Front Projection: LCC versus SC",
    fontsize=14
)

plt.xlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.ylabel(
    "Self-Consumption (SC)",
    fontsize=11
)

colorbar = plt.colorbar(projection_2)

colorbar.set_label(
    "Self-Sufficiency (SS)",
    fontsize=11
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ==========================================================
# 4. HÌNH CHIẾU SS – SC
#    MÀU BIỂU DIỄN LCC
# ==========================================================

idx_ss = np.argsort(SS)

plt.figure(figsize=(8, 6))

projection_3 = plt.scatter(
    SS,
    SC,
    c=LCC,
    s=45,
    edgecolors="black",
    linewidths=0.3
)

plt.plot(
    SS[idx_ss],
    SC[idx_ss],
    linewidth=1.2,
    alpha=0.7
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.2,
    label="Physical limit: SC = 1"
)

plt.axvline(
    x=1.0,
    linestyle="--",
    linewidth=1.2,
    label="Physical limit: SS = 1"
)

plt.title(
    "Pareto Front Projection: SS versus SC",
    fontsize=14
)

plt.xlabel(
    "Self-Sufficiency (SS)",
    fontsize=11
)

plt.ylabel(
    "Self-Consumption (SC)",
    fontsize=11
)

colorbar = plt.colorbar(projection_3)

colorbar.set_label(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("Done step 29")

import numpy as np
import matplotlib.pyplot as plt

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from matplotlib.lines import Line2D


# ==========================================================
# 1. LẤY TOÀN BỘ QUẦN THỂ CUỐI CÙNG
# ==========================================================

if result.pop is None:
    raise ValueError(
        "result.pop không tồn tại. "
        "Hãy kiểm tra lại kết quả trả về từ minimize()."
    )

F_population = np.asarray(
    result.pop.get("F"),
    dtype=float
)

X_population = np.asarray(
    result.pop.get("X"),
    dtype=float
)

if F_population.ndim != 2 or F_population.shape[1] != 3:
    raise ValueError(
        "F_population phải có shape (N, 3), "
        f"nhưng hiện tại là {F_population.shape}."
    )

# Trong bài toán của bạn:
# F = [-SS, -SC, LCC]

SS_population = -F_population[:, 0]
SC_population = -F_population[:, 1]
LCC_population = F_population[:, 2]


# ==========================================================
# 2. PHÂN LOẠI CÁC PARETO FRONT
# ==========================================================

front_indices = NonDominatedSorting().do(
    F_population
)

print("Số cá thể trong quần thể cuối:", len(F_population))
print("Số Pareto front:", len(front_indices))

for rank, indices in enumerate(front_indices, start=1):
    print(
        f"Front {rank}: {len(indices)} nghiệm"
    )


# Front 1: tập nghiệm không bị chi phối
front_1_idx = np.asarray(
    front_indices[0],
    dtype=int
)

F_front_1 = F_population[front_1_idx]

SS_front_1 = SS_population[front_1_idx]
SC_front_1 = SC_population[front_1_idx]
LCC_front_1 = LCC_population[front_1_idx]

X_front_1 = X_population[front_1_idx]

print("\nThông tin Pareto Front 1")
print("Số nghiệm:", len(front_1_idx))

print(
    f"SS : {SS_front_1.min():.6f} "
    f"→ {SS_front_1.max():.6f}"
)

print(
    f"SC : {SC_front_1.min():.6f} "
    f"→ {SC_front_1.max():.6f}"
)

print(
    f"LCC: {LCC_front_1.min():,.2f} "
    f"→ {LCC_front_1.max():,.2f}"
)


# ==========================================================
# 3. XÁC ĐỊNH CÁC NGHIỆM ĐẶC TRƯNG TRÊN FRONT 1
# ==========================================================

# SS lớn nhất
idx_max_ss_local = np.argmax(
    SS_front_1
)

# SC lớn nhất
idx_max_sc_local = np.argmax(
    SC_front_1
)

# LCC nhỏ nhất
idx_min_lcc_local = np.argmin(
    LCC_front_1
)


# ----------------------------------------------------------
# KNEE POINT
#
# Chuẩn hóa ba mục tiêu theo dạng bài toán cực tiểu:
# [-SS, -SC, LCC]
#
# Sau đó chọn điểm gần ideal point (0, 0, 0) nhất.
# ----------------------------------------------------------

F_min = F_front_1.min(axis=0)
F_max = F_front_1.max(axis=0)

F_range = F_max - F_min

# Tránh chia cho 0 nếu một mục tiêu không thay đổi
F_range[F_range == 0] = 1.0

F_front_1_normalized = (
    F_front_1 - F_min
) / F_range

distance_to_ideal = np.linalg.norm(
    F_front_1_normalized,
    axis=1
)

idx_knee_local = np.argmin(
    distance_to_ideal
)


# Hàm lấy tọa độ điểm đặc trưng
def representative_point(local_index):
    return (
        SS_front_1[local_index],
        SC_front_1[local_index],
        LCC_front_1[local_index]
    )


max_ss_point = representative_point(
    idx_max_ss_local
)

max_sc_point = representative_point(
    idx_max_sc_local
)

min_lcc_point = representative_point(
    idx_min_lcc_local
)

knee_point = representative_point(
    idx_knee_local
)


print("\nCác nghiệm đặc trưng")

print(
    "Maximum SS:",
    max_ss_point
)

print(
    "Maximum SC:",
    max_sc_point
)

print(
    "Minimum LCC:",
    min_lcc_point
)

print(
    "Knee point:",
    knee_point
)


# ==========================================================
# 4. HÀM VẼ CÁC FRONT BỊ CHI PHỐI
# ==========================================================

def plot_dominated_fronts_2d(
    x_values,
    y_values,
    max_fronts_to_show=3
):
    """
    Vẽ Front 2, Front 3,... dưới dạng nền mờ.
    Front 1 được vẽ riêng ở phía sau.
    """

    number_of_fronts = min(
        len(front_indices),
        max_fronts_to_show
    )

    # Vẽ từ front sâu hơn đến front gần hơn
    for rank in range(
        number_of_fronts - 1,
        0,
        -1
    ):
        indices = np.asarray(
            front_indices[rank],
            dtype=int
        )

        # Front 2 đậm hơn Front 3
        alpha_value = max(
            0.10,
            0.35 - 0.10 * (rank - 1)
        )

        point_size = max(
            12,
            25 - 5 * (rank - 1)
        )

        plt.scatter(
            x_values[indices],
            y_values[indices],
            s=point_size,
            alpha=alpha_value,
            edgecolors="none",
            label=f"Front {rank + 1}"
        )


def plot_dominated_fronts_3d(
    axis,
    max_fronts_to_show=3
):
    """
    Vẽ Front 2, Front 3,... trên đồ thị 3D.
    """

    number_of_fronts = min(
        len(front_indices),
        max_fronts_to_show
    )

    for rank in range(
        number_of_fronts - 1,
        0,
        -1
    ):
        indices = np.asarray(
            front_indices[rank],
            dtype=int
        )

        alpha_value = max(
            0.10,
            0.30 - 0.08 * (rank - 1)
        )

        point_size = max(
            10,
            22 - 4 * (rank - 1)
        )

        axis.scatter(
            SS_population[indices],
            SC_population[indices],
            LCC_population[indices],
            s=point_size,
            alpha=alpha_value,
            edgecolors="none"
        )


# ==========================================================
# 5. HÀM ĐÁNH DẤU CÁC NGHIỆM ĐẶC TRƯNG
# ==========================================================

def mark_representative_points_2d(
    x_front,
    y_front
):
    """
    Đánh dấu các nghiệm đặc trưng trên đồ thị 2D.
    """

    representative_points = [
        (
            idx_max_ss_local,
            "*",
            150,
            "Maximum SS"
        ),
        (
            idx_max_sc_local,
            "s",
            80,
            "Maximum SC"
        ),
        (
            idx_min_lcc_local,
            "D",
            75,
            "Minimum LCC"
        ),
        (
            idx_knee_local,
            "X",
            110,
            "Knee point"
        )
    ]

    for index, marker, size, label in representative_points:
        plt.scatter(
            x_front[index],
            y_front[index],
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=10
        )


def mark_representative_points_3d(axis):
    representative_points = [
        (
            idx_max_ss_local,
            "*",
            150,
            "Maximum SS"
        ),
        (
            idx_max_sc_local,
            "s",
            80,
            "Maximum SC"
        ),
        (
            idx_min_lcc_local,
            "D",
            75,
            "Minimum LCC"
        ),
        (
            idx_knee_local,
            "X",
            110,
            "Knee point"
        )
    ]

    for index, marker, size, label in representative_points:
        axis.scatter(
            SS_front_1[index],
            SC_front_1[index],
            LCC_front_1[index],
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=10
        )


# ==========================================================
# 6. ĐỒ THỊ PARETO 3D
# ==========================================================

fig = plt.figure(
    figsize=(9, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Front 2, Front 3...
plot_dominated_fronts_3d(
    ax,
    max_fronts_to_show=3
)

# Front 1
pareto_3d = ax.scatter(
    SS_front_1,
    SC_front_1,
    LCC_front_1,
    c=LCC_front_1,
    s=34,
    alpha=0.90,
    edgecolors="none",
    label="Pareto Front 1"
)

mark_representative_points_3d(
    ax
)

ax.set_title(
    "Three-Dimensional Pareto-Optimal Solutions",
    fontsize=14,
    pad=18
)

ax.set_xlabel(
    "Self-Sufficiency (SS)",
    fontsize=11,
    labelpad=10
)

ax.set_ylabel(
    "Self-Consumption (SC)",
    fontsize=11,
    labelpad=10
)

ax.set_zlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11,
    labelpad=10
)

colorbar = fig.colorbar(
    pareto_3d,
    ax=ax,
    shrink=0.72,
    pad=0.10
)

colorbar.set_label(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

ax.legend(
    loc="best",
    fontsize=8
)

plt.tight_layout()
plt.show()


# ==========================================================
# 7. HÌNH CHIẾU LCC – SS
#    MÀU BIỂU DIỄN SC
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

# Front 2, Front 3...
plot_dominated_fronts_2d(
    LCC_population,
    SS_population,
    max_fronts_to_show=3
)

# Front 1
projection_1 = plt.scatter(
    LCC_front_1,
    SS_front_1,
    c=SC_front_1,
    s=34,
    alpha=0.95,
    edgecolors="none",
    label="Pareto Front 1",
    zorder=5
)

mark_representative_points_2d(
    LCC_front_1,
    SS_front_1
)

plt.title(
    "Projection of Pareto-Optimal Solutions: LCC versus SS",
    fontsize=13
)

plt.xlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.ylabel(
    "Self-Sufficiency (SS)",
    fontsize=11
)

colorbar = plt.colorbar(
    projection_1
)

colorbar.set_label(
    "Self-Consumption (SC)",
    fontsize=11
)

plt.grid(
    alpha=0.15
)

plt.legend(
    fontsize=8,
    loc="best"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 8. HÌNH CHIẾU LCC – SC
#    MÀU BIỂU DIỄN SS
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

plot_dominated_fronts_2d(
    LCC_population,
    SC_population,
    max_fronts_to_show=3
)

projection_2 = plt.scatter(
    LCC_front_1,
    SC_front_1,
    c=SS_front_1,
    s=34,
    alpha=0.95,
    edgecolors="none",
    label="Pareto Front 1",
    zorder=5
)

mark_representative_points_2d(
    LCC_front_1,
    SC_front_1
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.1,
    label="Physical limit: SC = 1"
)

plt.title(
    "Projection of Pareto-Optimal Solutions: LCC versus SC",
    fontsize=13
)

plt.xlabel(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.ylabel(
    "Self-Consumption (SC)",
    fontsize=11
)

colorbar = plt.colorbar(
    projection_2
)

colorbar.set_label(
    "Self-Sufficiency (SS)",
    fontsize=11
)

plt.grid(
    alpha=0.15
)

plt.legend(
    fontsize=8,
    loc="best"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 9. HÌNH CHIẾU SS – SC
#    MÀU BIỂU DIỄN LCC
# ==========================================================

plt.figure(
    figsize=(8, 6)
)

plot_dominated_fronts_2d(
    SS_population,
    SC_population,
    max_fronts_to_show=3
)

projection_3 = plt.scatter(
    SS_front_1,
    SC_front_1,
    c=LCC_front_1,
    s=34,
    alpha=0.95,
    edgecolors="none",
    label="Pareto Front 1",
    zorder=5
)

mark_representative_points_2d(
    SS_front_1,
    SC_front_1
)

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.1,
    label="Physical limit: SC = 1"
)

plt.axvline(
    x=1.0,
    linestyle="--",
    linewidth=1.1,
    label="Physical limit: SS = 1"
)

plt.title(
    "Projection of Pareto-Optimal Solutions: SS versus SC",
    fontsize=13
)

plt.xlabel(
    "Self-Sufficiency (SS)",
    fontsize=11
)

plt.ylabel(
    "Self-Consumption (SC)",
    fontsize=11
)

colorbar = plt.colorbar(
    projection_3
)

colorbar.set_label(
    "Life-Cycle Cost (LCC)",
    fontsize=11
)

plt.grid(
    alpha=0.15
)

plt.legend(
    fontsize=8,
    loc="best"
)

plt.tight_layout()
plt.show()


# ==========================================================
# 10. XUẤT THÔNG SỐ THIẾT KẾ CỦA CÁC NGHIỆM ĐẶC TRƯNG
# ==========================================================

representative_designs = {
    "Maximum SS": X_front_1[idx_max_ss_local],
    "Maximum SC": X_front_1[idx_max_sc_local],
    "Minimum LCC": X_front_1[idx_min_lcc_local],
    "Knee point": X_front_1[idx_knee_local]
}

print("\nBiến thiết kế của các nghiệm đặc trưng")

for name, design_variables in representative_designs.items():
    print(f"\n{name}:")
    print(design_variables)
    
print("Done step 30")

# ==========================================================
# 31. XUẤT PARETO FRONT 1 RA FILE CSV
#     Mỗi dòng là một nghiệm: 8 biến thiết kế (công suất PV1..PV8)
#     + SS, SC, LCC. Cột "Representative" đánh dấu các nghiệm đặc trưng.
# ==========================================================

PARETO_CSV_PATH = "pareto_front_1.csv"

pareto_df = pd.DataFrame(
    X_front_1,
    columns=list(inputs.columns)
)

pareto_df["SS"] = SS_front_1
pareto_df["SC"] = SC_front_1
pareto_df["LCC"] = LCC_front_1

# Đánh dấu nghiệm đặc trưng (một nghiệm có thể mang nhiều nhãn)
representative_labels = {
    idx_max_ss_local: "Maximum SS",
    idx_max_sc_local: "Maximum SC",
    idx_min_lcc_local: "Minimum LCC",
    idx_knee_local: "Knee point"
}

pareto_df["Representative"] = ""

for local_index, label in representative_labels.items():
    current = pareto_df.at[local_index, "Representative"]
    pareto_df.at[local_index, "Representative"] = (
        label if current == "" else current + "; " + label
    )

# Sắp xếp theo LCC tăng dần cho dễ đọc
pareto_df = pareto_df.sort_values("LCC").reset_index(drop=True)

# utf-8-sig để Excel mở file không bị lỗi font
pareto_df.to_csv(
    PARETO_CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nĐã xuất {len(pareto_df)} nghiệm của Pareto Front 1 ra: "
    f"{os.path.abspath(PARETO_CSV_PATH)}"
)

print("Done step 31: export Pareto front to CSV")