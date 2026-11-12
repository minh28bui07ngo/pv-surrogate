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
            units= 4,
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

# [PATCH] tỉ lệ thô ngoài [0,1] = mạng "muốn" vi phạm bao nhiêu mẫu
r_raw = y_dataset_pred_raw[:, 2]
print(
    "Raw E_self_ratio ngoài [0,1] (trước khi ép):",
    int(np.sum((r_raw < 0) | (r_raw > 1))),
    "/", len(r_raw)
)

# Đưa về đại lượng vật lý hợp lệ
y_dataset_pred = to_physical(y_dataset_pred_raw)

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
# 6. KIỂM TRA RIÊNG ELECTRICITY:FACILITY
# ==========================================================

facility_true = y_dataset_true[:, 0]
facility_pred = y_dataset_pred[:, 0]
facility_error = facility_pred - facility_true

print("\n" + "=" * 80)
print("ELECTRICITY:FACILITY DIAGNOSTIC")
print("=" * 80)

print("True mean :", facility_true.mean())
print("Pred mean :", facility_pred.mean())

print("True std  :", facility_true.std(ddof=1))
print("Pred std  :", facility_pred.std(ddof=1))

print("Mean error:", facility_error.mean())
print("Std error :", facility_error.std(ddof=1))

print(
    "True-Pred correlation:",
    np.corrcoef(
        facility_true,
        facility_pred
    )[0, 1]
)


# ==========================================================
# 7. KIỂM TRA RÀNG BUỘC VẬT LÝ TRÊN DATASET
# ==========================================================

facility_pred = y_dataset_pred[:, 0]
produced_pred = y_dataset_pred[:, 1]
e_self_pred = y_dataset_pred[:, 2]

with np.errstate(
    divide="ignore",
    invalid="ignore"
):
    SS_pred_dataset = (
        e_self_pred / facility_pred
    )

    SC_pred_dataset = (
        e_self_pred / produced_pred
    )

finite_mask = (
    np.isfinite(SS_pred_dataset)
    & np.isfinite(SC_pred_dataset)
)

ss_violation = (
    finite_mask
    & (SS_pred_dataset > 1.0)
)

sc_violation = (
    finite_mask
    & (SC_pred_dataset > 1.0)
)

any_violation = (
    ss_violation
    | sc_violation
)

print("\n" + "=" * 80)
print("PHYSICAL CONSTRAINT CHECK ON FULL DATASET")
print("=" * 80)

print(f"Tổng số mẫu: {len(y_dataset_pred)}")

print(
    f"SS > 1: "
    f"{ss_violation.sum()} "
    f"({ss_violation.mean() * 100:.4f}%)"
)

print(
    f"SC > 1: "
    f"{sc_violation.sum()} "
    f"({sc_violation.mean() * 100:.4f}%)"
)

print(
    f"SS > 1 hoặc SC > 1: "
    f"{any_violation.sum()} "
    f"({any_violation.mean() * 100:.4f}%)"
)

print(
    f"SS/SC không hữu hạn: "
    f"{(~finite_mask).sum()}"
)

print(
    f"SS dự đoán: "
    f"{SS_pred_dataset[finite_mask].min():.6f} "
    f"→ "
    f"{SS_pred_dataset[finite_mask].max():.6f}"
)

print(
    f"SC dự đoán: "
    f"{SC_pred_dataset[finite_mask].min():.6f} "
    f"→ "
    f"{SC_pred_dataset[finite_mask].max():.6f}"
)

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
y_pred = to_physical(y_pred_raw)  # [PATCH]

# Giá trị thật
y_true = dataset_test[PHYS_COLS].to_numpy(dtype=float)  # [PATCH] E_self thật

# =====================================================
# 2. ĐÁNH GIÁ TỪNG OUTPUT
# =====================================================

output_names = [
    "Electricity:Facility",
    "ElectricityProduced:Facility",
    "E_self",
    "Annual_grid_cost(VND)"
]

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

facility = y_pred[:,0]
produced = y_pred[:,1]
e_self = y_pred[:,2]

SS = e_self / facility
SC = e_self / produced

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

error = y_pred[:,0] - y_true[:,0]

print(error.mean())
print(error.std())

print(y_true[:,0].mean())
print(y_pred[:,0].mean())

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
#    output (trên thang chuẩn hóa để 4 output so sánh được với nhau).
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

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

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

facility_index = 0

# ==========================================================
# 1. KIỂM TRA THỨ TỰ CỘT OUTPUT
# ==========================================================

print("=" * 80)
print("1. OUTPUT COLUMN ORDER")
print("=" * 80)

print("train_labels columns:")
print(list(train_labels.columns))

print("\ntest_labels columns:")
print(list(test_labels.columns))

print("\ntraining_labels index:")
print(list(training_labels.index))


# ==========================================================
# 2. THỐNG KÊ DÙNG ĐỂ CHUẨN HÓA OUTPUT
# ==========================================================

print("\n" + "=" * 80)
print("2. OUTPUT NORMALIZATION STATISTICS")
print("=" * 80)

print(training_labels)

facility_train_mean = training_labels.loc[
    "Electricity:Facility",
    "mean"
]

facility_train_std = training_labels.loc[
    "Electricity:Facility",
    "std"
]

print("\nFacility training mean:", facility_train_mean)
print("Facility training std :", facility_train_std)


# ==========================================================
# 3. KIỂM TRA Ở KHÔNG GIAN CHUẨN HÓA
# ==========================================================

print("\n" + "=" * 80)
print("3. NORMALIZED TRUE / PREDICTION")
print("=" * 80)

true_norm_facility = test_labels[
    "Electricity:Facility"
].to_numpy(dtype=float)

pred_norm_facility = y_pred_norm[:, facility_index]

print("True normalized mean:", true_norm_facility.mean())
print("Pred normalized mean:", pred_norm_facility.mean())

print("True normalized std :", true_norm_facility.std(ddof=1))
print("Pred normalized std :", pred_norm_facility.std(ddof=1))

print("Pred normalized min :", pred_norm_facility.min())
print("Pred normalized max :", pred_norm_facility.max())


# ==========================================================
# 4. KIỂM TRA Ở ĐƠN VỊ GỐC
# ==========================================================

print("\n" + "=" * 80)
print("4. ORIGINAL-SCALE TRUE / PREDICTION")
print("=" * 80)

true_facility = y_true[:, facility_index]
pred_facility = y_pred[:, facility_index]

print("True original mean:", true_facility.mean())
print("Pred original mean:", pred_facility.mean())

print("True original std :", true_facility.std(ddof=1))
print("Pred original std :", pred_facility.std(ddof=1))

print("True original min :", true_facility.min())
print("True original max :", true_facility.max())

print("Pred original min :", pred_facility.min())
print("Pred original max :", pred_facility.max())


# ==========================================================
# 5. KIỂM TRA SAI SỐ
# ==========================================================

error_facility = pred_facility - true_facility

print("\n" + "=" * 80)
print("5. FACILITY ERROR")
print("=" * 80)

print("Mean error       :", error_facility.mean())
print("Std error        :", error_facility.std(ddof=1))
print("Minimum error    :", error_facility.min())
print("Maximum error    :", error_facility.max())
print("Maximum abs error:", np.max(np.abs(error_facility)))


# ==========================================================
# 6. SO SÁNH 20 MẪU ĐẦU
# ==========================================================

compare_facility = pd.DataFrame({
    "True_norm": true_norm_facility,
    "Pred_norm": pred_norm_facility,
    "True_original": true_facility,
    "Pred_original": pred_facility,
    "Error_original": error_facility
})

print("\n" + "=" * 80)
print("6. TRUE VS PREDICTED")
print("=" * 80)

print(compare_facility.head(20))


# ==========================================================
# 7. TƯƠNG QUAN TRUE–PREDICTED VÀ R²
# ==========================================================

true_pred_corr = np.corrcoef(
    true_facility,
    pred_facility
)[0, 1]

facility_r2 = r2_score(
    true_facility,
    pred_facility
)

print("\n" + "=" * 80)
print("7. TRUE–PREDICTED RELATION")
print("=" * 80)

print("Pearson correlation:", true_pred_corr)
print("R²                 :", facility_r2)


# ==========================================================
# 8. SO SÁNH PHÂN BỐ DEVELOPMENT VÀ TEST
# ==========================================================

development_facility = dataset[
    "Electricity:Facility"
].to_numpy(dtype=float)

test_facility = dataset_test[
    "Electricity:Facility"
].to_numpy(dtype=float)

print("\n" + "=" * 80)
print("8. DEVELOPMENT / TEST DISTRIBUTION")
print("=" * 80)

print("\nDevelopment Facility")
print("Mean :", development_facility.mean())
print("Std  :", development_facility.std(ddof=1))
print("Min  :", development_facility.min())
print("Max  :", development_facility.max())

print("\nTest Facility")
print("Mean :", test_facility.mean())
print("Std  :", test_facility.std(ddof=1))
print("Min  :", test_facility.min())
print("Max  :", test_facility.max())

mean_shift = (
    development_facility.mean()
    - test_facility.mean()
)

print("\nMean difference:", mean_shift)

print(
    "Mean difference / Development std:",
    mean_shift / development_facility.std(ddof=1)
)

print("Done step 22")

corr = dataset.corr(numeric_only=True)

facility_corr = corr.loc[
    inputs.columns,
    "Electricity:Facility"
]

print(
    facility_corr
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print("Done step 23")

### Hàm chuẩn hóa dữ liệu đầu vào cho mô hình ANN 
def Standardization(population_PV):
    def norm_input(x):
        return (x - train_stats['mean'].to_numpy(dtype=float)) / train_stats['std'].to_numpy(dtype=float) ### Chuẩn hoá dữ liệu đưa dữ liệu về cùng 1 thang đo nhỏ hơn 
    inputs_ANN_norm = norm_input(population_PV)
    return inputs_ANN_norm
    
### Hàm chuyển thẳng từ Population_PV ra 3 thông số thực E self , Facility và Froduced 
def Prediction(population_PV):
    inputs_ANN_norm = Standardization(population_PV)
    ## Dự đoán output bằng mô hình ANN 
    output_ANN_norm = best_model.predict(
        inputs_ANN_norm,
        verbose=0
    )
    ### Truy xuất về dữ liệu thực 
    output_ANN = to_physical(unnorm_output(output_ANN_norm))  # [PATCH]
    return output_ANN
    
### Hàm LCC ,SS,SC
def calculate_LCC_SS_SC(population_PV,time = 20 , discount_rate = 0.1 ):
    ## Chuyển từ công suất PV định mức của mỗi tấm sang diện tích của mỗi tấm 
    area_PV = population_PV / 198 
     # Tổng diện tích 8 vùng PV của từng cá thể
    total_area_PV = np.sum(area_PV, axis=1)
    # Chi phí đầu tư ban đầu
    initial_PV_cost = (
        total_area_PV
        * 2_500_000
        * 1.15 )
    ### Lấy output đã được dự đoán và giải chuẩn hóa từ mô hình ANN 
    output_ANN = Prediction(population_PV)
    ### Tiền điện mua từ lưới qua từng năm mô phỏng  
    annual_grid_cost = output_ANN[:,3]
    # [PATCH] to_physical() đã ép cost >= 0 nên không cần raise ValueError

    ### Hàm tính chi phí vận hành , bảo dưỡng trong 20 năm 
    present_value_grid_cost = np.zeros(
    len(population_PV),
    dtype=float)
    ### Cộng dồn chí phí bảo trì qua 20 năm 
    for year in range(1, time + 1):
        present_value_grid_cost += (
            annual_grid_cost
            / (1 + discount_rate) ** year)
    ### Tính tổng LCC Cho các lần mô phỏng = CHÍ PHÍ đầu tư ban đầu + chi phí bảo trì sửa chữa qua 20 năm 
    LCC = initial_PV_cost + present_value_grid_cost
    ### Tính SS 
    SS = output_ANN[:,2]/np.maximum(output_ANN[:,0], 1.0)
    ### Tính SC
    SC = output_ANN[:,2]/np.maximum(output_ANN[:,1], 1.0)
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
    
# [PATCH] Nghiệm Pareto nằm trên biên bị ép (r_raw >= 1) là dấu hiệu
# NSGA-II đang khai thác sai số surrogate -> nên kiểm chứng bằng EnergyPlus.
raw_pareto = unnorm_output(
    best_model.predict(Standardization(X_pareto), verbose=0)
)
r_pareto_raw = raw_pareto[:, 2]
n_edge = int(np.sum(r_pareto_raw >= 1.0))
print(
    f"Nghiệm Pareto có raw ratio >= 1 (bị ép về 1): "
    f"{n_edge}/{len(r_pareto_raw)}"
)
print(
    f"Raw ratio lớn nhất: {r_pareto_raw.max():.4f}, "
    f"nhỏ nhất: {r_pareto_raw.min():.4f}"
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