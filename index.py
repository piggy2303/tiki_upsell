from multiprocessing import Process
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re
import pickle
from scipy import spatial

import requests
from bson.json_util import loads, dumps
import json
import argparse


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def euclidean_distance(a, b):
    dist = np.linalg.norm(a-b)
    return dist


def cosine_distance(a, b):
    result = 1 - spatial.distance.cosine(a, b)
    return result


df = pd.read_pickle("./df.pkl")
name_embeding = np.load("name_embeding.npy", allow_pickle=True)


def creat_graph(id):
    print("doing ./pair_id/pair_id_"+str(id)+".pkl")
    df_pair_id = pd.read_pickle("./pair_id/pair_id_"+str(id)+".pkl")

    for i in df_pair_id:
        df_item_from = df[df["id"] == i[0]]
        df_item_to = df[df["id"] == i[1]]
        name_embeding_id_from = np.where(
            (name_embeding[:, 0] == i[0]))[0][0]
        a = name_embeding[name_embeding_id_from][1][0]

        name_embeding_id_to = np.where((name_embeding[:, 0] == i[1]))[0][0]
        b = name_embeding[name_embeding_id_to][1][0]

        obj = {
            "from_id": i[0],
            "to_id": i[1],
            "euclidean_distance": float(euclidean_distance(a, b)),
            "cosine_distance": float(cosine_distance(a, b)),
            "same_category": bool((df_item_from["productset_group_name"].values == df_item_to["productset_group_name"].values)[0]),
            "same_brand_name": bool((df_item_from["brand.id"].values == df_item_to["brand.id"].values)[0]),
            "same_current_seller": bool((df_item_from["current_seller.id"].values == df_item_to["current_seller.id"].values)[0]),
            "more_expensive": bool((df_item_from["price"].values - df_item_to["price"].values)[0]),
            "more_discount": bool((df_item_from["discount"].values - df_item_to["discount"].values)[0]),
            "more_discount_rate": bool((df_item_from["discount_rate"].values - df_item_to["discount_rate"].values)[0]),
            "more_price_segment": int((df_item_from["price_segment"].values - df_item_to["price_segment"].values)[0]),
            "more_rating": int((df_item_from["rating_average"].values - df_item_to["rating_average"].values)[0]),
            "older": int((df_item_from["day_ago_created"].values - df_item_to["day_ago_created"].values)[0]),
            "more_review_count": int((df_item_from["review_count"].values - df_item_to["review_count"].values)[0]),
        }

# if __name__ == "__main__":  # confirms that the code is under main function
#     ids = [0, 1, 2]
#     procs = []
#     # proc = Process(target=creat_graph)  # instantiating without any argument
#     # procs.append(proc)
#     # proc.start()

#     # instantiating process with arguments
#     for id in ids:
#         # print(name)
#         proc = Process(target=creat_graph, args=(id,))
#         procs.append(proc)
#         proc.start()

#     # complete the processes
#     for proc in procs:
#         proc.join()
