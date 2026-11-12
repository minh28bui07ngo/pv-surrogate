import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#!pip install besos --user
### Nhóm thư viện thực hiện xử lý dữ liệu ( đọc file , lưu file dữ liệu mô phỏng , EDA ,tính toán (Numpy) 
import pandas as pd
import numpy as np

### NHóm thư viện Machine Learning / Deep Learning( Học máy và học sâu ) 
import tensorflow as tf   ### Bộ máy tính toán và huấn luyện mạng notron 
import tensorflow.keras as keras ## Keras là API ( GIAO THỨC ) Cấp cao của thư viện tensorflow
from tensorflow.keras import layers ### Dùng để tạo các lớp mạng 

### Besos Nền tảng mô phỏng, tối ưu hóa và mô hình hóa thay thế cho hệ thống năng lượng và công trình
from besos import eppy_funcs as ef ###Eppy là  Module của thư viện Besos dùng để chỉnh sửa file EnergyPlus.
import besos.sampling as sampling ### Module của thư viện Besos dùng để sinh các điểm mẫu 
from besos.problem import EPProblem ### Module mô tả bài toán tối ưu 
from besos.evaluator import EvaluatorEP ### Đây là cầu nối giữa Python và EnergyPlus.( sửa IDF/ chạy EnergyPlus /lấy kết quả)
from besos.parameters import wwr, RangeParameter,FieldSelector, Parameter ###
### Trong đó wwr là một hàm được viết sẵn để tự động tạo tham số WWR mà ko cần tự định nghĩa , tham số này chính là diện tích cửa số trên diện tích tường 
### RangeParameter là một lớp dùng để quy định biến được phép thay đổi trong khoảng nào  ( Tham số tạo khoảng giá trị ) , Besos Sampling sẽ lấy mẫu trong khoảng này 
### FieldSelector = Bộ chọn trường dữ liệu đây là đối tượng dùng để nói cho Besos biết Chính xác phải sửa chỗ nào trong file IDF 
### Parameter = Tham số thiết kế kết hợp giữa FieldSelector và RangeParameter

### Nhóm thư viện dùng để Vẽ biểu đồ , Theo dõi quá trình Train , Dừng sớm , Hiển thị kết quả đẹp hơn 
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import time
from dask.distributed import Client ### dòng lệnh này dùng để chạy nhiều mô phỏng Energy Plus song song nhằm tạo giữ liệu nhanh hơn,Nó không phải thư viện chính để huấn luyện mạng nơ-ron với nhiều tham số đầu vào cùng lúc; việc đó do TensorFlow đảm nhiệm. 

### Active-learning sampler: fits a small GP on PV1..PV8 -> objective, and
### only runs the most informative candidate points through EnergyPlus.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

print("Done step 0")

building = ef.get_building("D:/da2/in-new-fixed.idf")
building.view_model()

print("Done step 1")

insu1 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array1..PV1_c8a8c324', field_name='DC System Capacity')
insu2 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV2_3c2723f8', field_name='DC System Capacity')
insu3 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV3_6fa39fbf', field_name='DC System Capacity')
insu4 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV4_eac3702d', field_name='DC System Capacity')
insu5 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV5_f0496fd1', field_name='DC System Capacity')
insu6 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV6_87ace17d', field_name='DC System Capacity')
insu7 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV7_88b7b2d6', field_name='DC System Capacity')
insu8 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array..PV8_83328630', field_name='DC System Capacity')

print("Done step 2")

from besos.parameters import FieldSelector

PV10_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 1 Rated Electric Power Output"
)

PV20_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 2 Rated Electric Power Output"
)

PV30_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 3 Rated Electric Power Output"
)

PV40_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 4 Rated Electric Power Output"
)

PV50_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 5 Rated Electric Power Output"
)

PV60_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 6 Rated Electric Power Output"
)

PV70_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 7 Rated Electric Power Output"
)

PV80_selector = FieldSelector(
    class_name="ElectricLoadCenter:Generators",
    object_name="Model Load Center Distribution Generators",
    field_name="Generator 8 Rated Electric Power Output"
)

print("Done step 3")

from besos.parameters import FieldSelector

### Here we change all the external insulation of the building

############ insu1 (Shade_4ea2b0cd)
insu11x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_c8a8c324', field_name='Vertex 1 Xcoordinate')
insu11y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_c8a8c324', field_name='Vertex 1 Ycoordinate')
insu12x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_c8a8c324', field_name='Vertex 2 Xcoordinate')
insu12y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_c8a8c324', field_name='Vertex 2 Ycoordinate')

############ insu2 (Shade_dfde1d43)
insu23x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_3c2723f8', field_name='Vertex 3 Xcoordinate')
insu23y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_3c2723f8', field_name='Vertex 3 Ycoordinate')
insu24x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_3c2723f8', field_name='Vertex 4 Xcoordinate')
insu24y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_3c2723f8', field_name='Vertex 4 Ycoordinate')

############ insu3 (Shade_63033c6b)
insu33x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_6fa39fbf', field_name='Vertex 3 Xcoordinate')
insu33y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_6fa39fbf', field_name='Vertex 3 Ycoordinate')
insu34x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_6fa39fbf', field_name='Vertex 4 Xcoordinate')
insu34y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_6fa39fbf', field_name='Vertex 4 Ycoordinate')

############ insu4 (Shade_f6e3d36b)
insu41x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_eac3702d', field_name='Vertex 1 Xcoordinate')
insu41y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_eac3702d', field_name='Vertex 1 Ycoordinate')
insu42x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_eac3702d', field_name='Vertex 2 Xcoordinate')
insu42y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_eac3702d', field_name='Vertex 2 Ycoordinate')

############ insu5 (Shade_910726b5)
insu51x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_f0496fd1', field_name='Vertex 1 Xcoordinate')
insu51y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_f0496fd1', field_name='Vertex 1 Ycoordinate')
insu52x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_f0496fd1', field_name='Vertex 2 Xcoordinate')
insu52y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_f0496fd1', field_name='Vertex 2 Ycoordinate')

############ insu6 (Shade_6d7243a0)
insu63x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_87ace17d', field_name='Vertex 3 Xcoordinate')
insu63y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_87ace17d', field_name='Vertex 3 Ycoordinate')
insu64x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_87ace17d', field_name='Vertex 4 Xcoordinate')
insu64y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_87ace17d', field_name='Vertex 4 Ycoordinate')

############ insu7 (Shade_439c35f5)
insu73x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_88b7b2d6', field_name='Vertex 3 Xcoordinate')
insu73y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_88b7b2d6', field_name='Vertex 3 Ycoordinate')
insu74x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_88b7b2d6', field_name='Vertex 4 Xcoordinate')
insu74y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_88b7b2d6', field_name='Vertex 4 Ycoordinate')

############ insu8 (Shade_764d1a72)
insu81x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_83328630', field_name='Vertex 1 Xcoordinate')
insu81y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_83328630', field_name='Vertex 1 Ycoordinate')
insu82x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_83328630', field_name='Vertex 2 Xcoordinate')
insu82y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_83328630', field_name='Vertex 2 Ycoordinate')

print("Done step 4")

from besos import eppy_funcs as ef ###Eppy là  Module của thư viện Besos dùng để chỉnh sửa file EnergyPlus.
import besos.sampling as sampling ### Module của thư viện Besos dùng để sinh các điểm mẫu 
from besos.problem import EPProblem ### Module mô tả bài toán tối ưu 
from besos.evaluator import EvaluatorEP ### Đây là cầu nối giữa Python và EnergyPlus.( sửa IDF/ chạy EnergyPlus /lấy kết quả)
from besos.parameters import wwr, RangeParameter,FieldSelector, Parameter ###
###########################################################################
parameters_PV = [

    Parameter(
        value_descriptors=RangeParameter(198.000621, 73656.000079),
        name="PV1"
    ),

    Parameter(
        value_descriptors=RangeParameter( 198.000621, 73656.000079),
        name="PV2"
    ),

    Parameter(
        value_descriptors=RangeParameter(197.999648, 36827.998902),
        name="PV3"
    ),

    Parameter(
        value_descriptors=RangeParameter(197.999648,36827.998902),
        name="PV4"
    ),

    Parameter(
        value_descriptors=RangeParameter(990.000337,96227.999467),
        name="PV5"
    ),

    Parameter(
        value_descriptors=RangeParameter(990.000337,  96227.999467),
        name="PV6"
    ),

    Parameter(
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV7"
    ),

    Parameter(
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV8"
    )]
objective_PV = ['Electricity:Facility',"ElectricityProduced:Facility"] 
problem_PV = EPProblem(parameters_PV, objective_PV)

print("Done step 5")

# kHAI BÁO các đỉnh cố định 
T13 = np.array([32.3699842192161, 284.884604771589, 15.9797223425019])
T14 = np.array([29.7562875901585, 289.375550098717, 18.9797223425018])
T21 = np.array([29.7562875901585, 289.375550098717, 18.9797223425018])
T22 = np.array([27.1425909611011, 293.866495425845, 15.9797223425019])
T31 = np.array([-15.456514790388, 269.233900289854, 19.2508104401878])
T32 = np.array([-10.96556946326, 271.847596918912, 16.2508104401879])
T43 = np.array([-19.9474601175161, 266.620203660797, 16.2508104401879])
T44 = np.array([-15.456514790388, 269.233900289854, 19.2508104401878])
T53 = np.array([10.6317395667062, 321.72788941284, 16.1256341107898])
T54 = np.array([8.01804293764854, 326.218834739968, 19.1256341107899])
T61 = np.array([8.01804293764854, 326.218834739968, 19.1256341107899])
T62 = np.array([5.40434630859102, 330.709780067096, 16.12563411079])
T71 = np.array([28.667026036303, 317.732331179357, 12.8178503081051])
T72 = np.array([20.0199026617534, 312.770889077372, 8.16904942895734])
T83 = np.array([37.3141494108522, 322.693773281343, 8.16904942895742])
T84 = np.array([28.667026036303, 317.732331179357, 12.8178503081051])

print("Done step 8")

import numpy as np
n1 = np.array( [ 8.64282831e-01 , 5.03006151e-01, -2.5439184366719718e-15])
n2 = np.array([-8.64282831e-01 ,-5.03006151e-01 , 8.46811206e-15])
n3 = np.array([-0.50300615,  0.86428283 , 0       ])
n4 = np.array([ 0.50300615, -0.86428283 , 0         ])
n5 = np.array([0.86428283, 0.50300615 ,0        ])
n6 = np.array( [-0.86428283 ,-0.50300615  ,0        ])
n7 = np.array( [ 0.49766779 ,-0.86736773 , 0        ])
n8 = np.array([-0.49766779,  0.86736773 , 0      ])

print("Done step 9")

### Setup hàm chuyển tham số định nghĩa sang tham số toạ độ 
import numpy as np
import pandas as pd

# =========================
# OFFSET FUNCTION
# =========================
def apply_offset(T_base, n, d, sign=1):
    """
    T_new = T_base ± (d · n)
    """
    T_base = np.asarray(T_base)
    n = np.asarray(n).reshape(1, 3)
    d = np.asarray(d).reshape(-1, 1)

    return T_base + sign * (d * n)

# =========================================================================
# EXPAND FUNCTION  (wraps your original "Done step 6" through "Done step 10"
# logic into a callable, so it can run on a batch of ANY size -- not just
# the fixed 200-row inputs_PV from before. Nothing in the math below changed
# from your original script; it's the same area_PV -> inputs_d ->
# apply_offset -> inputs -> inputs_all pipeline, just parameterized.)
# =========================================================================
fixed_length = [6, 6, 6, 6, 6, 6, 11, 11]

def expand_inputs(inputs_PV):
    """inputs_PV: DataFrame with columns PV1..PV8 (any number of rows).
    Returns inputs_all: DataFrame with the same 48 columns your original
    script built (T11_X..T82_Y, PV1..PV8, PV10..PV80)."""

    # Chia công suất cho 198 W/m² để ra diện tích
    area_PV = inputs_PV / 198
    inputs_d = area_PV.div(fixed_length, axis=1)
    inputs_d.columns = [f"d{i}" for i in range(1, 9)]

    d1 = inputs_d["d1"].to_numpy()
    d2 = inputs_d["d2"].to_numpy()
    d3 = inputs_d["d3"].to_numpy()
    d4 = inputs_d["d4"].to_numpy()
    d5 = inputs_d["d5"].to_numpy()
    d6 = inputs_d["d6"].to_numpy()
    d7 = inputs_d["d7"].to_numpy()
    d8 = inputs_d["d8"].to_numpy()

    # =========================
    # GEOMETRY UPDATE
    # =========================
    T11 = apply_offset(T14, n1, d1, sign=-1)
    T12 = apply_offset(T13, n1, d1, sign=-1)

    T23 = apply_offset(T22, n2, d2, sign=+1)
    T24 = apply_offset(T21, n2, d2, sign=+1)

    T33 = apply_offset(T32, n3, d3, sign=+1)
    T34 = apply_offset(T31, n3, d3, sign=+1)

    T41 = apply_offset(T44, n4, d4, sign=-1)
    T42 = apply_offset(T43, n4, d4, sign=-1)

    T51 = apply_offset(T54, n5, d5, sign=-1)
    T52 = apply_offset(T53, n5, d5, sign=-1)

    T63 = apply_offset(T62, n6, d6, sign=+1)
    T64 = apply_offset(T61, n6, d6, sign=+1)

    T73 = apply_offset(T72, n7, d7, sign=+1)
    T74 = apply_offset(T71, n7, d7, sign=+1)

    T81 = apply_offset(T84, n8, d8, sign=-1)
    T82 = apply_offset(T83, n8, d8, sign=-1)

    # =========================
    # ONLY TAKE X, Y (DROP Z)
    # =========================
    inputs = pd.DataFrame(np.hstack([
        T11[:, :2], T12[:, :2],
        T23[:, :2], T24[:, :2],
        T33[:, :2], T34[:, :2],
        T41[:, :2], T42[:, :2],
        T51[:, :2], T52[:, :2],
        T63[:, :2], T64[:, :2],
        T73[:, :2], T74[:, :2],
        T81[:, :2], T82[:, :2],
    ]), columns=[
        "T11_X","T11_Y", "T12_X","T12_Y",
        "T23_X","T23_Y", "T24_X","T24_Y",
        "T33_X","T33_Y", "T34_X","T34_Y",
        "T41_X","T41_Y", "T42_X","T42_Y",
        "T51_X","T51_Y", "T52_X","T52_Y",
        "T63_X","T63_Y", "T64_X","T64_Y",
        "T73_X","T73_Y", "T74_X","T74_Y",
        "T81_X","T81_Y", "T82_X","T82_Y",
    ], index=inputs_PV.index)

    # Lần 1 giữ nguyên tên cột
    PV1 = inputs_PV.copy()
    # Lần 2 đổi tên cột
    PV2 = inputs_PV.copy()
    PV2.columns = [f"{col}0" for col in PV2.columns]

    return pd.concat([inputs, PV1, PV2], axis=1)

print("Done step 6-10 (now wrapped in expand_inputs)")

parameters = [

    # =====================
    # insu1 (Shade_4ea2b0cd)
    # =====================
    Parameter(insu11x, value_descriptors=RangeParameter(-1e6, 1e6), name='T11_X'),
    Parameter(insu11y, value_descriptors=RangeParameter(-1e6, 1e6), name='T11_Y'),
    Parameter(insu12x, value_descriptors=RangeParameter(-1e6, 1e6), name='T12_X'),
    Parameter(insu12y, value_descriptors=RangeParameter(-1e6, 1e6), name='T12_Y'),

    # =====================
    # insu2 (Shade_dfde1d43)
    # =====================
    Parameter(insu23x, value_descriptors=RangeParameter(-1e6, 1e6), name='T23_X'),
    Parameter(insu23y, value_descriptors=RangeParameter(-1e6, 1e6), name='T23_Y'),
    Parameter(insu24x, value_descriptors=RangeParameter(-1e6, 1e6), name='T24_X'),
    Parameter(insu24y, value_descriptors=RangeParameter(-1e6, 1e6), name='T24_Y'),

    # =====================
    # insu3 (Shade_63033c6b)
    # =====================
    Parameter(insu33x, value_descriptors=RangeParameter(-1e6, 1e6), name='T33_X'),
    Parameter(insu33y, value_descriptors=RangeParameter(-1e6, 1e6), name='T33_Y'),
    Parameter(insu34x, value_descriptors=RangeParameter(-1e6, 1e6), name='T34_X'),
    Parameter(insu34y, value_descriptors=RangeParameter(-1e6, 1e6), name='T34_Y'),

    # =====================
    # insu4 (Shade_f6e3d36b)
    # =====================
    Parameter(insu41x, value_descriptors=RangeParameter(-1e6, 1e6), name='T41_X'),
    Parameter(insu41y, value_descriptors=RangeParameter(-1e6, 1e6), name='T41_Y'),
    Parameter(insu42x, value_descriptors=RangeParameter(-1e6, 1e6), name='T42_X'),
    Parameter(insu42y, value_descriptors=RangeParameter(-1e6, 1e6), name='T42_Y'),

    # =====================
    # insu5 (Shade_910726b5)
    # =====================
    Parameter(insu51x, value_descriptors=RangeParameter(-1e6, 1e6), name='T51_X'),
    Parameter(insu51y, value_descriptors=RangeParameter(-1e6, 1e6), name='T51_Y'),
    Parameter(insu52x, value_descriptors=RangeParameter(-1e6, 1e6), name='T52_X'),
    Parameter(insu52y, value_descriptors=RangeParameter(-1e6, 1e6), name='T52_Y'),

    # =====================
    # insu6 (Shade_6d7243a0)
    # =====================
    Parameter(insu63x, value_descriptors=RangeParameter(-1e6, 1e6), name='T63_X'),
    Parameter(insu63y, value_descriptors=RangeParameter(-1e6, 1e6), name='T63_Y'),
    Parameter(insu64x, value_descriptors=RangeParameter(-1e6, 1e6), name='T64_X'),
    Parameter(insu64y, value_descriptors=RangeParameter(-1e6, 1e6), name='T64_Y'),

    # =====================
    # insu7 (Shade_439c35f5)
    # =====================
    Parameter(insu73x, value_descriptors=RangeParameter(-1e6, 1e6), name='T73_X'),
    Parameter(insu73y, value_descriptors=RangeParameter(-1e6, 1e6), name='T73_Y'),
    Parameter(insu74x, value_descriptors=RangeParameter(-1e6, 1e6), name='T74_X'),
    Parameter(insu74y, value_descriptors=RangeParameter(-1e6, 1e6), name='T74_Y'),

    # =====================
    # insu8 (Shade_764d1a72)
    # =====================
    Parameter(insu81x, value_descriptors=RangeParameter(-1e6, 1e6), name='T81_X'),
    Parameter(insu81y, value_descriptors=RangeParameter(-1e6, 1e6), name='T81_Y'),
    Parameter(insu82x, value_descriptors=RangeParameter(-1e6, 1e6), name='T82_X'),
    Parameter(insu82y, value_descriptors=RangeParameter(-1e6, 1e6), name='T82_Y'),
#####
    Parameter(insu1,
        value_descriptors=RangeParameter(198.000621, 73656.000079),
        name="PV1"
    ),

    Parameter(insu2,
        value_descriptors=RangeParameter( 198.000621, 73656.000079),
        name="PV2"
    ),

    Parameter(insu3,
        value_descriptors=RangeParameter(197.999648, 36827.998902),
        name="PV3"
    ),

    Parameter(insu4,
        value_descriptors=RangeParameter(197.999648,36827.998902),
        name="PV4"
    ),

    Parameter(insu5,
        value_descriptors=RangeParameter(990.000337,96227.999467),
        name="PV5"
    ),

    Parameter(insu6,
        value_descriptors=RangeParameter(990.000337,  96227.999467),
        name="PV6"
    ),

    Parameter(insu7,
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV7"
    ),

    Parameter(insu8,
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV8"
    ),
##########################
    Parameter(PV10_selector,
        value_descriptors=RangeParameter(198.000621, 73656.000079),
        name="PV10"
    ),

    Parameter(PV20_selector,
        value_descriptors=RangeParameter( 198.000621, 73656.000079),
        name="PV20"
    ),

    Parameter(PV30_selector,
        value_descriptors=RangeParameter(197.999648, 36827.998902),
        name="PV30"
    ),

    Parameter(PV40_selector,
        value_descriptors=RangeParameter(197.999648,36827.998902),
        name="PV40"
    ),

    Parameter(PV50_selector,
        value_descriptors=RangeParameter(990.000337,96227.999467),
        name="PV50"
    ),

    Parameter(PV60_selector,
        value_descriptors=RangeParameter(990.000337,  96227.999467),
        name="PV60"
    ),

    Parameter(PV70_selector,
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV70"
    ),

    Parameter(PV80_selector,
        value_descriptors=RangeParameter(198.001915, 69695.998978),
        name="PV80"
    )
]
print("Done step 11")
from besos.objectives import MeterReader, time_series_values

objective = [
    MeterReader(
        key_name="Electricity:Facility",
        frequency="Hourly",
        func=time_series_values,
        name="Electricity:Facility"
    ),
    MeterReader(
        key_name="ElectricityProduced:Facility",
        frequency="Hourly",
        func=time_series_values,
        name="ElectricityProduced:Facility"
    )
]

print("Done step 13")

problem = EPProblem(parameters, objective)


evaluator = EvaluatorEP(
    problem,
    building,
    epw ="D:/da2/VNM_SVN_Nha.Be.488940_TMYx.epw",
    progress_bar=True,
    out_dir="outputdirectory")

print("Done step 14")
# =========================================================================
# ACTIVE LEARNING LOOP  (replaces: inputs_PV = sampling.dist_sampler(...,200)
# + PV1/PV2/inputs_all concat + evaluator.df_apply(inputs_all, processes=6))
#
# Instead of one blind 200-row batch, this runs a small seed batch, fits a
# Gaussian Process on PV1..PV8 -> a scalar objective, then repeatedly picks
# only the most informative next points to actually run through EnergyPlus.
# =========================================================================
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

PV_NAMES = [f"PV{i}" for i in range(1, 9)]
 
# Starting points, not guarantees. Rule of thumb for a GP in D dims with a
# smooth, noise-free objective (EnergyPlus is deterministic per input):
# seed with ~8-12x D, then let validation error decide when to stop.
N_SEED = 100           # ~10x the 8 true dimensions -- enough for the GP's
                       # first length-scale estimate to be trustworthy
N_VALID = 60           # held-out set, run ONCE, never used to fit the GP --
                       # this is what actually measures "quality"
N_PER_ROUND = 30       # real EnergyPlus runs added each active round
MAX_ROUNDS = 50        # safety cap: 80 + 15*20 = 380, still under 500
PATIENCE = 2           # stop after this many rounds with no real improvement
MIN_IMPROVEMENT = 0.005 # relative RMSE improvement below this counts as "no improvement"
POOL_SIZE = 4000       # candidate points scored per round (free, no EnergyPlus)
 
 
def summarize_objective(outputs_hourly_batch):
    """Scalar score used ONLY to steer sampling: annual net grid draw.
    Full hourly profiles are still kept untouched in outputs_hourly_all."""
    elec = outputs_hourly_batch["Electricity:Facility"].apply(np.sum)
    pv = outputs_hourly_batch["ElectricityProduced:Facility"].apply(np.sum)
    return (elec - pv).to_numpy()
 
 
def fit_gp(X, y):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) \
        + WhiteKernel(noise_level=1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)
    return gp
 
 
def pick_next_batch(gp, candidate_pool, n_pick):
    """Pure-exploration acquisition: run the points the GP is least sure
    about first, so each new EnergyPlus run teaches the surrogate the most."""
    _, std = gp.predict(candidate_pool, return_std=True)
    top_idx = np.argsort(std)[-n_pick:]
    return candidate_pool[top_idx]
 
 
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
 
 
print(f"Số tiến trình sử dụng: 6")
t1 = time.time()
 
inputs_all_batches = []
outputs_hourly_batches = []
 
# --- Held-out validation set: run ONCE, real EnergyPlus, never trained on.
#     This is the actual yardstick -- "quality" means low RMSE here, not a
#     round count. It also means the validation runs count against your
#     total budget, same as any other real EnergyPlus call. ---
valid_PV = sampling.dist_sampler(sampling.lhs, problem_PV, N_VALID)
valid_inputs_all = expand_inputs(valid_PV)
valid_outputs = evaluator.df_apply(valid_inputs_all, processes=6)
X_valid = valid_PV[PV_NAMES].to_numpy()
y_valid = summarize_objective(valid_outputs)
print(f"Validation set: {N_VALID} real EnergyPlus runs (held out, never trained on).")
 
# --- Seed round: small, blind, space-filling ---
seed_PV = sampling.dist_sampler(sampling.lhs, problem_PV, N_SEED)
seed_inputs_all = expand_inputs(seed_PV)
seed_outputs = evaluator.df_apply(seed_inputs_all, processes=6)
 
inputs_all_batches.append(seed_inputs_all)
outputs_hourly_batches.append(seed_outputs)
X = seed_PV[PV_NAMES].to_numpy()
y = summarize_objective(seed_outputs)
 
pd.concat(inputs_all_batches).to_pickle("inputs_train.pkl")
pd.concat(outputs_hourly_batches).to_pickle("outputs_train.pkl")
 
gp = fit_gp(X, y)
best_rmse = rmse(gp.predict(X_valid), y_valid)
rounds_without_improvement = 0
print(f"Round 0 (seed): {len(X)} real EnergyPlus runs. Validation RMSE: {best_rmse:.2f}. "
      f"Saved inputs2.pkl / outputs2.pkl.")
 
# --- Active rounds: keep going until validation RMSE stops improving ---
for round_i in range(1, MAX_ROUNDS + 1):
    pool_PV = sampling.dist_sampler(sampling.lhs, problem_PV, POOL_SIZE)
    pool = pool_PV[PV_NAMES].to_numpy()
 
    next_batch = pick_next_batch(gp, pool, n_pick=N_PER_ROUND)
    next_PV = pd.DataFrame(next_batch, columns=PV_NAMES)
 
    next_inputs_all = expand_inputs(next_PV)
    next_outputs = evaluator.df_apply(next_inputs_all, processes=6)
 
    inputs_all_batches.append(next_inputs_all)
    outputs_hourly_batches.append(next_outputs)
    X = np.vstack([X, next_batch])
    y = np.concatenate([y, summarize_objective(next_outputs)])
 
    inputs_all = pd.concat(inputs_all_batches).reset_index(drop=True)
    outputs_hourly = pd.concat(outputs_hourly_batches).reset_index(drop=True)
    inputs_all.to_pickle("inputs_train.pkl")
    outputs_hourly.to_pickle("outputs_train.pkl")
 
    gp = fit_gp(X, y)
    current_rmse = rmse(gp.predict(X_valid), y_valid)
    relative_improvement = (best_rmse - current_rmse) / best_rmse if best_rmse > 0 else 0
 
    print(f"Round {round_i}: +{N_PER_ROUND} runs, total {len(X)} real EnergyPlus runs. "
          f"Validation RMSE: {current_rmse:.2f} "
          f"({relative_improvement:+.1%} vs. best so far). Saved inputs_train.pkl / outputs_train.pkl.")
 
    if relative_improvement > MIN_IMPROVEMENT:
        best_rmse = current_rmse
        rounds_without_improvement = 0
    else:
        rounds_without_improvement += 1
        if rounds_without_improvement >= PATIENCE:
            print(f"Validation RMSE stopped improving for {PATIENCE} rounds -- stopping. "
                  f"Additional real EnergyPlus runs are unlikely to meaningfully help "
                  f"a GP surrogate of this shape; this is the point of diminishing "
                  f"returns for THIS objective on THIS building, not a guess.")
            break
 
t2 = time.time()
print(f"Thời gian mô phỏng: {(t2 - t1) / 60:.2f} phút")
print(f"Done. Total real EnergyPlus runs: {len(X) + N_VALID} "
      f"({len(X)} training + {N_VALID} validation), final validation RMSE: {best_rmse:.2f}.")
 
# inputs2.pkl / outputs2.pkl are already saved incrementally above --
# safe to Ctrl+C after any round and still have a usable dataset.