"""Import some packages"""
import os
import time, random
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""                       Print some stuff                                 """
"""----------------------------------------------------------------------"""

# Print list in a vertical form
def print_list(list):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    for l in list:
        print(l)
    print("++++++++++++++++++++++++++++++++++++++++++++")

# Print dictionary in a vertical form
def print_dict(dict):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    for k, v in dict.items():
        print("key:", k, "   value:", v)
    print("++++++++++++++++++++++++++++++++++++++++++++")

# Print something with a log marker
def print_with_log(info):
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print(info)
    print("++++++++++++++++++++++++++++++++++++++++++++")

# Print log marker
def print_log():
    print("++++++++++++++++++++++++++++++++++++++++++++")

"""                           File Storage                                 """
"""----------------------------------------------------------------------"""

# Save results to a json file
def save_to_json(info, filename, encoding='UTF-8'):
    with open(filename, "w", encoding=encoding) as f:
        json.dump(info, f, indent=2, separators=(',', ':'))

# Read from a json file
def load_from_json(filename):
    with open(filename, encoding='utf-8') as f:
        info = json.load(f)
    return info

# Store as a npy file
def save_to_npy(info, filename):
    np.save(filename, info, allow_pickle=True)

# Read from a npy file
def load_from_npy(filename):
    info = np.load(filename, allow_pickle=True)
    return info

# Save results to a txt file
def log_to_txt(contexts=None, filename="save.txt", mark=False, encoding='UTF-8', add_n=False):
    f = open(filename, "a", encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c) + " | " + str(contexts[c]) + "\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts, list):
            tmp = ""
            for c in contexts:
                if add_n:
                    tmp += str(c) + "\n"
                else:
                    tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)

    f.close()

# Read lines from a txt file
def load_from_txt(filename, encoding="utf-8"):
    f = open(filename, 'r', encoding=encoding)
    contexts = f.readlines()
    return contexts

"""                           Dictionary Transformation                   """
"""----------------------------------------------------------------------"""

# Exchange keys and values
def dict_k_v_exchange(dict):
    tmp = {}
    for key, value in dict.items():
        tmp[value] = key
    return tmp

# Convert 2D array to dictionary
def d2array_to_dict(d2array):
    # Input: N x 2 list
    # Output: dict
    dict = {}
    for item in d2array:
        if item[0] not in dict.keys():
            dict[item[0]] = [item[1]]
        else:
            dict[item[0]].append(item[1])
    return dict

"""                             Plotting                                  """
"""----------------------------------------------------------------------"""

# Plot 3D image
def visual_3d_points(list, color=True):
    """
    :param list: N x (dim +1)
    N is the number of points
    dim is the dimension of the input data
    1 is the category, i.e. the color for visualization, when and only when color is True
    """
    list = np.array(list)
    if color:
        data = list[:, :4]
        label = list[:, -1]
    else:
        data = list
        label = None

    # Dimensionality reduction with PCA
    pca = PCA(n_components=3, whiten=True).fit(data)
    data = pca.transform(data)

    # Define the axes
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    if label is not None:
        color = label
    else:
        color = "blue"
    ax1.scatter3D(np.transpose(data)[0], np.transpose(data)[1], np.transpose(data)[2], c=color)  # Draw scatter plot

    plt.show()

"""                           Utility Tools                              """
"""----------------------------------------------------------------------"""

# Calculate the number of occurrences of elements in an array
def count_list(lens):
    dict = {}
    for key in lens:
        dict[key] = dict.get(key, 0) + 1
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    print_list(dict)
    return dict

# List addition with weights w1 and w2
def list_add(list1, list2, w1=1, w2=1):
    return [l1 * w1 + l2 * w2 for (l1, l2) in zip(list1, list2)]
