import os
import subprocess
import helpers.sumoClass as sumoClass

SUMO_PATH = os.getenv("SUMO_HOME")


def create_taz_from_zone_shapes(
    scenario_folder="../san_francisco/",
    zones_file="san_francisco_zones.json",
    taz_id_attr="i",
    output_file="taZes.taz.xml",
):
    """
    Create TAZ (Traffic Analysis Zones) from zone shapes.

    Parameters:
    - scenario_folder (str): Path to the scenario folder.
    - zones_file (str): Zone file in JSON format.
    - taz_id_attr (str): Attribute for TAZ identification.
    - output_file (str): Output TAZ XML file.
    """
    os.chdir(scenario_folder)
    subprocess.run(
        "netconvert --sumo-net-file network.net.xml"
        + " --plain-output-prefix network --proj.plain-geo",
        shell=True,
    )

    # The above command generates files in the current working directory.
    # You may need to find a way to move the files to the target directory.

    subprocess.run(
        "python " + SUMO_PATH + "/tools/xml/xml2csv.py " + "network.edg.xml", shell=True
    )

    os.chdir("../src/")
    geo = sumoClass.Sumo_Network(
        path_network_csv=scenario_folder + "network.net.csv",
        path_edge_csv=scenario_folder + "network.edg.csv",
    )

    taz = sumoClass.Create_TAZs(
        path_sumo_network_geojson=scenario_folder + "network.geojson",
        path_zones_geojson=scenario_folder + zones_file,
    )

    t = taz.get_tazs(attribute=taz_id_attr)
    taz.write_tazs(output_path=scenario_folder + output_file)


if __name__ == "__main__":
    # Example usage for creating TAZs
    create_taz_from_zone_shapes()
