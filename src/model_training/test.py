import os

def model_name(path):
    model_name_arr = os.listdir(path)
    print(model_name_arr)
    i = 1
    name = f"model{i}.keras"
    while name in model_name_arr:
        name = f"model{i}.keras"
        i += 1

    return name[:(5+len(str(i)))]

print(model_name("../models"))