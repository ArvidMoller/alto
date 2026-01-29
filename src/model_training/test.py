import os
import numpy as np

# def model_name(path):
#     model_name_arr = os.listdir(path)
#     print(model_name_arr)
#     i = 1
#     name = f"model{i}.keras"
#     while name in model_name_arr:
#         name = f"model{i}.keras"
#         i += 1

#     return name[:(5+len(str(i)))]

# print(model_name("../models"))


dataset = np.array([[[0, 10], [20, 30]], [[40, 50], [60, 70]]])
predicted_frame = np.array([[8, 9], [10, 11]])

low = -1
high = 1

dataset = dataset / (100/(high - low)) - abs(low)
print(dataset)

dataset = (dataset + abs(low)) * (100/(high - low))
print(dataset)