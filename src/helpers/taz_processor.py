import pandas as pd
from collections import Counter
import os

input_edge_csv = os.environ.get("input_edge_csv")
input_taz_csv = os.environ.get("input_taz_csv")
output_taz_csv = os.environ.get("output_taz_csv")

df_network = pd.read_csv(input_edge_csv, sep=";")
list_edges = []
for node, degree in Counter(df_network.edge_from).most_common():
    if degree == 1:
        list_edges.append(df_network[df_network.edge_from == node]["edge_id"].values[0])
df_taz = pd.read_csv(input_taz_csv, sep=";")
taz_id_keep = []
for k, (taz_id, taz_edges) in enumerate(zip(df_taz.taz_id, df_taz.taz_edges)):
    edges = [i for i in taz_edges.split(" ")]
    for k in edges:
        if k in list_edges:
            taz_id_keep.append(taz_id)
            continue
df_taz_new = df_taz[df_taz.taz_id.isin(taz_id_keep)]
df_taz_new.to_csv(output_taz_csv, index=None)
