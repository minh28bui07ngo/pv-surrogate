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

print("Done step 0")

building = ef.get_building("in.idf")

import re

# 1) Inspect what eppy actually holds for the SpaceList
sl = building.idfobjects["SPACELIST"][0]
print(sl.Name, len(sl.obj))
print([i for i, v in enumerate(sl.obj) if v == ""])   # blank entries, if any

# 2) Rebuild the list from the original file text
txt = open("in.idf", encoding="utf-8", errors="ignore").read()
m = re.search(r"\nSpaceList,\s*ProgramType_493f593f,(.*?);", txt, re.S)
names = [re.sub(r"!.*", "", l).strip().rstrip(",;")
         for l in m.group(1).split("\n") if l.strip()]
sl.obj = ["SPACELIST", "ProgramType_493f593f"] + names
print(len(names), "names restored")

# 3) Check that the round trip is clean before running anything in parallel
building.saveas("D:/da2/besos_roundtrip.idf")

building.view_model()

print("Done step 1")

insu1 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array1..PV1_e1f4a045', field_name='DC System Capacity')
insu2 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 2..PV2_e5eb77ca', field_name='DC System Capacity')
insu3 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 3..PV3_b6de6a39', field_name='DC System Capacity')
insu4 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 4..PV4_f19716dd', field_name='DC System Capacity')
insu5 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 5..PV5_61524d1b', field_name='DC System Capacity')
insu6 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 6..PV6_b4f9bcfa', field_name='DC System Capacity')
insu7 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 7..PV7_0f7e0449', field_name='DC System Capacity')
insu8 = FieldSelector(class_name="Generator:PVWatts", object_name='Rooftop Array 8..PV8_da885889', field_name='DC System Capacity')

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
insu11x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_e1f4a045', field_name='Vertex 1 Xcoordinate')
insu11y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_e1f4a045', field_name='Vertex 1 Ycoordinate')
insu12x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_e1f4a045', field_name='Vertex 2 Xcoordinate')
insu12y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV1_e1f4a045', field_name='Vertex 2 Ycoordinate')

############ insu2 (Shade_dfde1d43)
insu23x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_e5eb77ca', field_name='Vertex 3 Xcoordinate')
insu23y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_e5eb77ca', field_name='Vertex 3 Ycoordinate')
insu24x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_e5eb77ca', field_name='Vertex 4 Xcoordinate')
insu24y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV2_e5eb77ca', field_name='Vertex 4 Ycoordinate')

############ insu3 (Shade_63033c6b)
insu33x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_b6de6a39', field_name='Vertex 3 Xcoordinate')
insu33y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_b6de6a39', field_name='Vertex 3 Ycoordinate')
insu34x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_b6de6a39', field_name='Vertex 4 Xcoordinate')
insu34y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV3_b6de6a39', field_name='Vertex 4 Ycoordinate')

############ insu4 (Shade_f6e3d36b)
insu41x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_f19716dd', field_name='Vertex 1 Xcoordinate')
insu41y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_f19716dd', field_name='Vertex 1 Ycoordinate')
insu42x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_f19716dd', field_name='Vertex 2 Xcoordinate')
insu42y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV4_f19716dd', field_name='Vertex 2 Ycoordinate')

############ insu5 (Shade_910726b5)
insu51x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_61524d1b', field_name='Vertex 1 Xcoordinate')
insu51y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_61524d1b', field_name='Vertex 1 Ycoordinate')
insu52x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_61524d1b', field_name='Vertex 2 Xcoordinate')
insu52y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV5_61524d1b', field_name='Vertex 2 Ycoordinate')

############ insu6 (Shade_6d7243a0)
insu63x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_b4f9bcfa', field_name='Vertex 3 Xcoordinate')
insu63y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_b4f9bcfa', field_name='Vertex 3 Ycoordinate')
insu64x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_b4f9bcfa', field_name='Vertex 4 Xcoordinate')
insu64y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV6_b4f9bcfa', field_name='Vertex 4 Ycoordinate')

############ insu7 (Shade_439c35f5)
insu73x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_0f7e0449', field_name='Vertex 3 Xcoordinate')
insu73y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_0f7e0449', field_name='Vertex 3 Ycoordinate')
insu74x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_0f7e0449', field_name='Vertex 4 Xcoordinate')
insu74y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV7_0f7e0449', field_name='Vertex 4 Ycoordinate')

############ insu8 (Shade_764d1a72)
insu81x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_da885889', field_name='Vertex 1 Xcoordinate')
insu81y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_da885889', field_name='Vertex 1 Ycoordinate')
insu82x = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_da885889', field_name='Vertex 2 Xcoordinate')
insu82y = FieldSelector(class_name='Shading:Building:Detailed', object_name='PV8_da885889', field_name='Vertex 2 Ycoordinate')

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

# VVV SỐ MẪU CẦN VVV
inputs_PV = sampling.dist_sampler(sampling.lhs, problem_PV,200)
inputs_PV.head(250)

print("Done step 6")

# Chia công suất cho 198 W/m² để ra diện tích
area_PV = inputs_PV / 198

# Chiều dài cạnh cố định:
# PV1 → PV6: 6 m
# PV7 → PV8: 11 m
fixed_length = [6, 6, 6, 6, 6, 6, 11, 11]

# Lấy chiều d còn lại
inputs_d = area_PV.div(fixed_length, axis=1)

# Đổi tên cột thành d1 ... d8
inputs_d.columns = [f"d{i}" for i in range(1, 9)]

print(inputs_d) ### Đây là tham số thể hiện chiều dài của cạnh thay đổi được ( chiều dài của tấm PV hình chữ nhật) 

print("Done step 7")

# ---- Replacement for "step 8" (fixed vertices T..) and "step 9" (normals n1..n8) ----
# Reads the fixed vertices and ridge directions straight from the NEW in.idf,
# so they always match the model. Variable names are the same as before.
 
_txt = open("in.idf", encoding="utf-8", errors="ignore").read()
 
def _vertices(pv):
    m = re.search(rf"Shading:Building:Detailed,\s*PV{pv}_\w+,(.*?)Vertex 4[^\n]*", _txt, re.S)
    nums = re.findall(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[,;]\s*!-\s*X,Y,Z Vertex", m.group(0))
    return np.array(nums, float)            # rows = vertex 1..4
 
_V = {p: _vertices(p) for p in range(1, 9)}
 
# fixed vertices (same names/meaning as the old script: T<pv><vertex>)
T13, T14 = _V[1][2], _V[1][3]
T21, T22 = _V[2][0], _V[2][1]
T31, T32 = _V[3][0], _V[3][1]
T43, T44 = _V[4][2], _V[4][3]
T53, T54 = _V[5][2], _V[5][3]
T61, T62 = _V[6][0], _V[6][1]
T71, T72 = _V[7][0], _V[7][1]
T83, T84 = _V[8][2], _V[8][3]
 
# unit direction n such that  moved = fixed + sign*d*n  (sign as in apply_offset calls)
def _n(pv, moved_idx, fixed_idx, sign):
    vec = _V[pv][moved_idx - 1] - _V[pv][fixed_idx - 1]
    return sign * vec / np.linalg.norm(vec)
 
n1 = _n(1, 1, 4, -1); n2 = _n(2, 3, 2, +1); n3 = _n(3, 3, 2, +1); n4 = _n(4, 1, 4, -1)
n5 = _n(5, 1, 4, -1); n6 = _n(6, 3, 2, +1); n7 = _n(7, 3, 2, +1); n8 = _n(8, 1, 4, -1)

print("Done step 9")

### Setup hàm chuyển tham số định nghĩa sang tham số toạ độ 
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


# =========================
# INPUT
# =========================
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
    "T11_X","T11_Y",
    "T12_X","T12_Y",

    "T23_X","T23_Y",
    "T24_X","T24_Y",

    "T33_X","T33_Y",
    "T34_X","T34_Y",

    "T41_X","T41_Y",
    "T42_X","T42_Y",

    "T51_X","T51_Y",
    "T52_X","T52_Y",

    "T63_X","T63_Y",
    "T64_X","T64_Y",

    "T73_X","T73_Y",
    "T74_X","T74_Y",

    "T81_X","T81_Y",
    "T82_X","T82_Y"
])
print(inputs.head(5))

print("Done step 10")

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

import pandas as pd

# Lần 1 giữ nguyên tên cột
PV1 = inputs_PV.copy()

# Lần 2 đổi tên cột
PV2 = inputs_PV.copy()
PV2.columns = [f"{col}0" for col in PV2.columns]

# Ghép theo chiều cột
inputs_all = pd.concat([inputs, PV1, PV2], axis=1)

print(inputs_all.head())

print("Done step 12")

from direct_sim import simulate_df
outputs_hourly = simulate_df(inputs_all, workers=6)

### Lưu kết quả input và output vào 2 file inputs.plk và outputs.plk 
inputs_all.to_pickle("inputs_test.pkl")
outputs_hourly.to_pickle("outputs_test.pkl")

print("Done step 16")