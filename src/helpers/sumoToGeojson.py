import json
import pandas as pd
import subprocess
import os

SUMO_HOME = os.environ.get("SUMO_HOME")
OUTPUT_PATH = "../demos/"
INPUT_PATH_NETWORK = "network.net.xml"
INPUT_PATH = "network.edg.xml"
OUTPUT_FILE = OUTPUT_PATH + "network.geojson"

# Change the current working directory to the output path
os.chdir(OUTPUT_PATH)

# Convert SUMO network to plain files
subprocess.run(
    "netconvert --sumo-net-file "
    + OUTPUT_PATH
    + INPUT_PATH_NETWORK
    + " --plain-output-prefix network --proj.plain-geo",
    shell=True,
)

# Convert the edge file to CSV
subprocess.run(
    "python " + SUMO_HOME + "/tools/xml/xml2csv.py " + OUTPUT_PATH + INPUT_PATH,
    shell=True,
)

# Change the working directory back to the source directory
os.chdir("../src/")

# Read the edge data from the CSV file
df_edge = pd.read_csv(
    OUTPUT_PATH + INPUT_PATH[:-3] + "csv", sep=";", error_bad_lines=False
)

# Filter out rows with missing data
df_edge = df_edge[df_edge["edge_id"].notna()]
df_edge = df_edge[df_edge["edge_shape"].notna()]
df_edge = df_edge[df_edge["edge_type"].notna()]

# Reset the index
df_edge.reset_index(inplace=True, drop=True)

# Select relevant columns
df_edge = df_edge[["edge_id", "edge_shape", "edge_speed", "edge_type"]]

# Separate edge information
edge_id = df_edge["edge_id"]
shapes = df_edge["edge_shape"]
edge_type = df_edge["edge_type"]

# Convert shapes to a format suitable for GeoJSON
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

# Create a list of features for GeoJSON
features = []
for i, edge in enumerate(zip(edge_id, conv_shapes, edge_type)):
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": edge[1]},
        "properties": {"id": edge[0], "type": edge[2]},
    }
    features.append(feature)

# Create a GeoJSON FeatureCollection
feature_collection = {"type": "FeatureCollection", "features": features}

# Write the GeoJSON data to a file
with open(OUTPUT_FILE, "w") as f:
    f.write(json.dumps(feature_collection, sort_keys=False, indent=4))

print("Finished!")
