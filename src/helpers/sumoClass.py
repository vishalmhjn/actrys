import pandas as pd
import geopandas as gpd
import geojson
import csv, json
from geojson import Feature, FeatureCollection, Point
class Sumo_Network:
    def __init__(self,
                path_network_csv="../data/paris_auto.net.csv",
                path_edge_csv="../data/plainnet.edg.csv"):
        # XML2CSV
        self.path_edge = path_edge_csv
        # XML2CSV
        self.path_network_csv = path_network_csv

    def write_edges(self):
        """function to write the number of lanes and length of an edge, for placing the detectors
        shst match ../data/sumo_network.geojson --out=sumo_network.geojson --snap-intersections --follow-line-direction"""
        df_net = pd.read_csv(self.path_network_csv, sep=';', error_bad_lines=False)
        temp = df_net[['edge_id','lane_id','lane_length']].dropna()
        temp = temp.drop_duplicates(subset='edge_id')
        edge_len = temp[['edge_id', 'lane_length']]
        edge_len.reset_index(inplace=True, drop=True)

        df_edge = pd.read_csv(self.path_edge, sep=";", error_bad_lines=False)
        df_edge = df_edge[df_edge['edge_id'].notna()]
        # df_edge = df_edge[df_edge['edge_shape'].notna()]
        df_edge_lanes = df_edge[["edge_id", "edge_numLanes"]]

        edge_len['edge_id'] = edge_len['edge_id'].astype(str)
        df_edge_lanes['edge_id'] = df_edge_lanes['edge_id'].astype(str)

        df_edge_length = pd.merge(df_edge_lanes, edge_len, on="edge_id")
        df_edge_length = df_edge_length.drop_duplicates(subset='edge_id')
        # df_edge_length.to_csv("../scenario_munich/temp_check.csv", index=None)
        return df_edge_length, df_edge_lanes

    def sumo_net_to_geojson(self, output_path="../data/sumo_network.geojson"):
        """convert sumo network to geojson for later use in shared streets and
        TAZs"""
        df_edge = pd.read_csv(self.path_edge, sep=";")
        
        df_edge = df_edge[df_edge['edge_id'].notna()]
        df_edge = df_edge[df_edge['edge_type'].notna()]
        try:
            df_edge = df_edge[df_edge['edge_shape'].notna()]
        except Exception as e:
            print(e)
            raise("Check first show of *.edg.xml is it has edge_shape attribute")
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
        with open(output_path, 'w') as f:
            f.write(json.dumps(d, sort_keys=False, indent=4))

class Create_TAZs:
    def __init__(self,
                path_sumo_network_geojson='../data/sumo_network.geojson',
                # this geojson should be the manually created as sumolib doesnot caontains link_type
                path_zones_geojson='../data/paris_communes.json'):
        self.path_network = path_sumo_network_geojson
        self.path_zones = path_zones_geojson
        self.network, self.zones = self.read_geojson()

    def read_geojson(self):
        with open(self.path_zones) as f:
            zones =  geojson.load(f)
        with open(self.path_network) as f:
            network =  geojson.load(f)
        return network, zones

    def get_tazs(self, attribute="MOVEMENT_ID"):
        """function to get the taz file using spatial join"""
        self.osm_attr = attribute
        zones = gpd.GeoDataFrame(self.zones['features'])
        # MOVEMENT_ID is the column of the id of the shape object in the shape
        # or geojson file. This should be changed as per the file. This has nothing to do with the
        # UBER data
        zones['id'] = zones['properties'].apply(lambda x: x[self.osm_attr])
        zones.drop(columns = ['type', 'properties'], inplace =True)

        edges = gpd.GeoDataFrame(self.network['features'])
        edges['edge_id'] = edges['properties'].apply(lambda x: x['id'])
        edges['link_type'] = edges['properties'].apply(lambda x: x['type'])
        edges.drop(columns = ['type', 'properties'], inplace =True)
        taz = gpd.sjoin(edges, zones, how="inner", op='within')

        return taz

    def write_tazs(self,
                    output_path="../data/taZes.taz.xml"):
        taz = self.get_tazs(attribute=self.osm_attr)
        a = """<tazs>"""
        b = ""
        ##### here is a big assumption on the creation of the trips
        ##### only sceondary and tertiary roads are used
        ##### this could be changed based on land-use density based trip creation
        for i in taz[taz['link_type'].isin(['highway.secondary', 'highway.tertiary'])].id.unique(): #'highway.primary' use when lower roads are missing
            temp = taz[taz.id==i]['edge_id']
            x = (list(temp))
            b = b+"""<taz edges="""+"\""+str(" ".join(x))+"\" id=\""+str(i+1)+"\""+"""/>""" #+1 in the zone if for tomtom data
        c = """</tazs>"""
        file_text = a+b+c
        with open(output_path, "w") as f:
            f.write(file_text)
            f.close()

if __name__ == "__main__":


    # convert sumo network_to_geojson
    # geo = Sumo_Network()
    # geo.sumo_net_to_geojson()

    # create TAZs
    taz = Create_TAZs('../scenario_munich/network.geojson', '../data/munich_zones_manual/manual_zones_munich_2.geojson')
    t = taz.get_tazs("id")
    taz.write_tazs("../msm_scenario_munich/new_taZes.taz.xml")