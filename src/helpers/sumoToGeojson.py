import csv, json
from geojson import Feature, FeatureCollection, Point
import pandas as pd
import subprocess
import os


SUMO_HOME = os.environ.get("SUMO_HOME")
# KeyError: 'edge_shape' on this error, go to plainnet.xml and check
# if the first row has edge_shape attribute

OUTPUT_PATH = "../demos/"#"../san_francisco/"
INPUT_PATH_NETWORK = "network.net.xml" #"network.net.xml"
INPUT_PATH = "network.edg.xml"
OUTPUT_FILE = OUTPUT_PATH+"network.geojson" #'../data/sumo_network.geojson'

os.chdir(OUTPUT_PATH)
# convert network to plain files 
subprocess.run("netconvert --sumo-net-file "+OUTPUT_PATH+INPUT_PATH_NETWORK+\
				" --plain-output-prefix network --proj.plain-geo", shell=True)

# convert edge file to csv
subprocess.run("python "+SUMO_HOME+"/tools/xml/xml2csv.py "+OUTPUT_PATH+INPUT_PATH, shell=True)

os.chdir("../src/")
df_edge = pd.read_csv(OUTPUT_PATH+INPUT_PATH[:-3]+"csv", sep=";", error_bad_lines=False) # use error_bad_lines=False if error
df_edge = df_edge[df_edge['edge_id'].notna()]
df_edge = df_edge[df_edge['edge_shape'].notna()]
df_edge = df_edge[df_edge['edge_type'].notna()]
df_edge.reset_index(inplace=True, drop=True)
df_edge = df_edge[['edge_id', 'edge_shape','edge_speed', 'edge_type']]

edge_id = df_edge['edge_id']
shapes = df_edge['edge_shape']
edge_type = df_edge['edge_type']
conv_shapes = []
for shape in shapes:
    shape_list = shape.split(" ")
    edge_shape = []
    for point in shape_list:
        lat = float(point.split(",")[0])
        lon = float(point.split(",")[1])
        node = [lat, lon]
        edge_shape.append(node)
    conv_shapes.append(edge_shape)

li = []
for i, edge in enumerate(zip(edge_id, conv_shapes, edge_type)):
    d = {}
    d['type'] = 'Feature'
    d['geometry'] = {'type': 'LineString','coordinates': edge[1]}
    d['properties'] = {'id': edge[0], 'type': edge[2]}
    li.append(d)
d = {}
d['type'] = 'FeatureCollection'
d['features'] = li
with open(OUTPUT_FILE, 'w') as f:
    f.write(json.dumps(d, sort_keys=False, indent=4))
print("Finished!")