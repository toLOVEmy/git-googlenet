import math

# 余弦退火参数
eta_min = 0.0000375
eta_max = 0.00015
T_max = 60
T_cur = 40

# 计算当前学习率
eta_t = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_max)) / 2
print(f"Epoch {T_cur} 时的学习率: {eta_t}")
