import subprocess, os
import pandas as pd
from collections import Counter

scenario_folder = "../san_francisco/temp/"
SUMO_HOME = os.getenv("SUMO_HOME")

if scenario_folder == "../san_francisco/temp/":
    # Define additional text and OD file names for the San Francisco scenario
    extra_text = """<taz>...</taz>"""  # Abbreviated for brevity
    od_file_names = "OD_file_SF_5.0_6.0.txt, OD_file_SF_6.0_7.0.txt, OD_file_SF_7.0_8.0.txt, OD_file_SF_8.0_9.0.txt, OD_file_SF_9.0_10.0.txt"
elif scenario_folder == "../scenario_munich/temp/":
    # Define additional text and OD file names for the Munich scenario
    extra_text = """<taz>...</taz>"""  # Abbreviated for brevity
    od_file_names = "MR_5.0_6.0.txt, MR_6.0_7.0.txt, MR_7.0_8.0.txt, MR_8.0_9.0.txt, MR_9.0_10.0.txt"
else:
    raise ValueError("The scenario folder is not covered")


# Function to generate trips
def generate_trips(taz_file="newtaZes.taz.xml"):
    os.chdir(scenario_folder)
    subprocess.run(
        "od2trips -n " + taz_file + " -d " + od_file_names + " -o trips.trips.xml",
        shell=True,
    )
    # Additional steps to process trips
    subprocess.run(
        "duarouter -n ../network.net.xml -r trips.trips.xml --ignore-errors -o routes.rou.xml --error-log errors.txt",
        shell=True,
    )
    subprocess.run(
        "python "
        + SUMO_HOME
        + "/tools/purgatory/route2trips.py routes.rou.xml > validated_trips.trips.xml",
        shell=True,
    )
    os.chdir("../../src/")


# Function to get disconnected edges
def get_disconnected_edges(error_file=scenario_folder + "errors.txt"):
    impossible_trips = open(error_file, "r")
    Lines = impossible_trips.readlines()
    new_lines = []
    count = 0
    for line in Lines:
        # just a clever way to find the unique rows of the mismatched edges
        if line[-7:] == "found.\n":
            new_lines.append(line)

    error = pd.DataFrame(new_lines)
    error["source"] = error[0].apply(lambda x: x.split(" ")[5])
    error["destination"] = error[0].apply(lambda x: x.split(" ")[8])
    error["mis_match"] = error.apply(lambda x: x.source + x.destination, axis=1)

    # Get the edges where the number of disconnected trips is more than a threshold
    threshold = 3
    error_source = [
        i[0].split("''")[0][1:]
        for i in Counter(error.mis_match).most_common()
        if i[1] > threshold
    ]
    error_destination = [
        i[0].split("''")[1][:-1]
        for i in Counter(error.mis_match).most_common()
        if i[1] > threshold
    ]
    return error_source, error_destination


# Function to count the length of trip files
def get_length_trip_files():
    trip_file = open(scenario_folder + "validated_trips.trips.xml", "r")
    Lines = trip_file.readlines()
    count = 0
    for line in Lines:
        count += 1
    return count - 10  # Subtracting XML headers


# Function to filter TAZs
def filter_tazs(remove_source, remove_destination, output_path="newtaZes.taz.xml"):
    os.chdir(scenario_folder)
    subprocess.run(
        "python " + SUMO_HOME + "/tools/xml/xml2csv.py " + output_path, shell=True
    )

    taz = pd.read_csv(output_path[:-3] + "csv", sep=";")
    taz_int = taz[taz["taz_edges"].notna()]["taz_edges"]
    taz_ext = list(taz[taz["taz_edges"].isna()]["tazSource_id"])
    taz_ext.extend(list(taz[taz["taz_edges"].isna()]["tazSink_id"]))
    taz_ext = [i for i in taz_ext if str(i) != "nan"]

    taz_id = taz[taz["taz_edges"].notna()]["taz_id"]

    taz_dict = dict()

    for i, j in enumerate(zip(taz_int, taz_id)):
        taz_new_edges = []
        temp_int_links = j[0].split(" ")

        if len(temp_int_links) > 3:  # At least three edges in the TAZ
            for k in temp_int_links:
                if k in remove_destination:
                    continue
                elif k in remove_source:
                    continue
                else:
                    taz_new_edges.append(k)
        else:
            taz_new_edges = temp_int_links
        taz_dict[str(j[1])] = taz_new_edges
    os.chdir("../../src/")
    return taz_dict


# Function to write TAZs
def write_tazs(taz_dict, output_path=scenario_folder + "newtaZes.taz.xml"):
    a = """<tazs>"""
    b = ""
    for i, j in enumerate(zip(list(taz_dict.keys()), list(taz_dict.values()))):
        x = j[1]
        b = (
            b
            + """<taz edges="""
            + '"'
            + str(" ".join(x))
            + '" id="'
            + str(j[0])
            + '"'
            + """/>"""
        )
    global extra_text
    c = """</tazs>"""
    file_text = a + b + extra_text + c
    with open(output_path, "w") as f:
        f.write(file_text)
        f.close()


if __name__ == "__main__":
    expected_trips = 450000
    generate_trips(taz_file="../taZes.taz.xml")

    counter = 0
    while counter < 5:
        remove_source, remove_destination = get_disconnected_edges()
        new_taz_dict = filter_tazs(remove_source, remove_destination)
        write_tazs(new_taz_dict)
        generate_trips()
        counter += 1

    print("Done")
