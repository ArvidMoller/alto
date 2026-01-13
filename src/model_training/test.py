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


dataset = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
predicted_frame = np.array([[8, 9], [10, 11]])

dataset = list(dataset)
dataset.append(predicted_frame)
dataset = np.delete(np.array(dataset), 0, axis=0)
print(dataset)