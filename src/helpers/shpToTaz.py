import sys
import os
import helpers.sumoClass as sumoClass
import subprocess
SUMO_PATH = os.getenv("SUMO_HOME")

def create_taz_from_zone_shapes(scenario_folder = "../san_francisco/",
                                zones_file="san_francisco_zones.json",
                                taz_id_attr="i",
                                output_file="taZes.taz.xml"):

    os.chdir(scenario_folder)
    subprocess.run("netconvert --sumo-net-file network.net.xml" +\
                    " --plain-output-prefix network --proj.plain-geo" ,shell=True)

    # the above command generates files in the pwd. need to finid a way  to move the  files to target
    # directory

    subprocess.run("python " + SUMO_PATH+ "/tools/xml/xml2csv.py " + \
                    "network.edg.xml", shell=True)

    os.chdir("../src/")
    geo = sumoClass.Sumo_Network(path_network_csv= scenario_folder+"network.net.csv",
                        path_edge_csv= scenario_folder+"network.edg.csv")
    
    taz = sumoClass.Create_TAZs(path_sumo_network_geojson=scenario_folder+'network.geojson',
                    path_zones_geojson= scenario_folder+zones_file)
    
    t = taz.get_tazs(attribute=taz_id_attr)
    # print(t)
    taz.write_tazs(output_path=scenario_folder+output_file)

if __name__ == "__main__":

    # for Munich
    create_taz_from_zone_shapes()

    # deprecated
    # subprocess.run("netconvert --sumo-net-file ../../scenario_munich/network.net.xml \
    # 				--plain-output-prefix network --proj.plain-geo" ,shell=True)

    # subprocess.run("python " + SUMO_PATH+ "/tools/xml/xml2csv.py ../../scenario_munich/network.edg.xml", shell=True)
    
    # geo.sumo_net_to_geojson(output_path="../../scenario_munich/network.geojson")



    # taz = sumo_class.Create_TAZs(path_sumo_network_geojson='../../scenario_munich/network.geojson',
    # 				path_zones_geojson='../../scenario_munich/munich_5_zones.geojson')
    # t = taz.get_tazs(attribute="id")
    # print(t)
    # taz.write_tazs(output_path="../../scenario_munich/taZes5.taz.xml")

    # for san francisco
    # create_taz_from_zone_shapes(scenario_folder="../../san_francisco/",
    # 							zones_file="san_francisco_censustracts.json",
    # 							taz_id_attr="MOVEMENT_ID",
    # 							output_file="taZes.taz.xml")
