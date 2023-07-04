##### this is used to filter the trips from the TAX file recursively
##### until the output of the do2trips generates similar trips as those
##### intended by removing the links which have disconnected from other links

import subprocess, os
import pandas as pd
from collections import Counter

scenario_folder = "../san_francisco/temp/"
SUMO_HOME = os.getenv("SUMO_HOME")

if scenario_folder == "../san_francisco/temp/":
    extra_text = """		<taz id="23_ext"> 
                <tazSource id="256665043#1" weight="20"/>
                <tazSource id="396955006" weight="20"/>
                <tazSource id="-417092384" weight="20"/>
                <tazSource id="-397127316#1" weight="20"/>
                <tazSource id="619289207" weight="20"/>
        
                <tazSink id="256665045#0" weight="13"/> 
                <tazSink id="429574317" weight="12"/>
                <tazSink id="91797305" weight="13"/>
                <tazSink id="417092392" weight="12"/>
                <tazSink id="123456285" weight="12"/>
                <tazSink id="397127316#0 " weight="12.5"/>
                <tazSink id="123867343" weight="12.5"/>
            </taz> 
        
            <taz id="56_ext">
                <tazSource id="309857692" weight="100"/> 
                <tazSink id="-309857692" weight="100"/> 
            </taz>
        
            <taz id="96_ext">
                <tazSource id="537838948" weight="100"/> 
                <tazSink id="595194543" weight="100"/> 
            </taz>
        
            <taz id="105_ext">
                <tazSource id="417385658" weight="50"/>
                <tazSource id="-397121529" weight="50"/> 
                <tazSink id="8942305" weight="50"/>
                <tazSink id="397161062" weight="50"/>
            </taz>
        
        
            <taz id="158_ext" >
                <tazSource id="120813417.201" weight="100"/> 
                <tazSink id="50690291" weight="100"/> 
            </taz>
        
        
            <taz id="182_ext" >
                <tazSource id="513704144#1" weight="100"/>
                <tazSink id="931324852" weight="100"/> 
            </taz>"""
    od_file_names = (
        "../demand/OD_file_SF_5.0_6.0.txt,"
        + "../demand/OD_file_SF_6.0_7.0.txt,../demand/OD_file_SF_7.0_8.0.txt,"
        + "../demand/OD_file_SF_8.0_9.0.txt,../demand/OD_file_SF_9.0_10.0.txt"
    )
elif scenario_folder == "../scenario_munich/temp/":
    extra_text = """<taz id="52814612"> 
                <tazSource id="127664113" weight="100"/> 
                <tazSink id="142289785" weight="100"/> 
            </taz> 
            <taz id="554321029"> 
                <tazSource id="234086364" weight="100"/> 
                <tazSink id="60657601#1" weight="100"/> 
            </taz> 
            <taz id="586888382"> 
                <tazSource id="144691956" weight="100"/> 
                <tazSink id="32848595" weight="100"/> 
            </taz> 
            <taz id="586900421"> 
                <tazSource id="4478922" weight="100"/> 
                <tazSink id="29264205" weight="100"/> 
            </taz> 
            <taz id="774028738"> 
                <tazSource id="24405955" weight="100"/> 
                <tazSink id="256297923" weight="100"/> 
            </taz> 
            <taz id="783095006"> 
                <tazSource id="153080104" weight="100"/> 
                <tazSink id="325882164" weight="100"/> 
            </taz> 
            <taz id="783176672"> 
                <tazSource id="280342521" weight="70"/> 
                <tazSource id="3345068" weight="30"/> 
                <tazSink id="280273997" weight="70"/> 
                <tazSink id="75719622" weight="30"/> 
            </taz> 
            <taz id="783176708"> 
                <tazSource id="194605224" weight="100"/> 
                <tazSink id="365480197" weight="100"/> 
            </taz> 
            <taz id="784358748"> 
                <tazSource id="280260315" weight="100"/> 
                <tazSink id="-280260315" weight="100"/> 
            </taz> 
            <taz id="819400248"> 
                <tazSource id="325030860" weight="100"/> 
                <tazSink id="3995738" weight="100"/> 
            </taz>"""
    od_file_names = (
        "../true_demand/MR_5.0_6.0.txt,"
        + "../true_demand/MR_6.0_7.0.txt,../true_demand/MR_7.0_8.0.txt,"
        + "../true_demand/MR_8.0_9.0.txt,../true_demand/MR_9.0_10.0.txt"
    )
else:
    raise ("The scenario folder is not covered")


def generate_trips(taz_file="newtaZes.taz.xml"):
    os.chdir(scenario_folder)
    subprocess.run(
        "od2trips -n " + taz_file + " -d " + od_file_names + " -o trips.trips.xml",
        shell=True,
    )
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


def get_disconnected_edges(error_file=scenario_folder + "errors.txt"):
    impossible_trips = open(error_file, "r")
    Lines = impossible_trips.readlines()
    new_lines = []
    count = 0
    for line in Lines:
        #### just a clever way to find the unique rows of the mismatched edges
        if line[-7:] == "found.\n":
            new_lines.append(line)

    error = pd.DataFrame(new_lines)
    error["source"] = error[0].apply(lambda x: x.split(" ")[5])
    error["destination"] = error[0].apply(lambda x: x.split(" ")[8])
    error["mis_match"] = error.apply(lambda x: x.source + x.destination, axis=1)

    ### Get the edges where the number of disconnected trips is more than 10
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


def get_length_trip_files():
    trip_file = open(scenario_folder + "validated_trips.trips.xml", "r")
    Lines = trip_file.readlines()
    count = 0
    for line in Lines:
        count += 1
    return count - 10  # subtracting xml headers


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

        if len(temp_int_links) > 3:  # at least three edges in the taz
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


def write_tazs(taz_dict, output_path=scenario_folder + "newtaZes.taz.xml"):
    a = """<tazs>"""
    b = ""
    # print(taz_dict)
    for i, j in enumerate(zip(list(taz_dict.keys()), list(taz_dict.values()))):
        x = j[1]
        # print(x)
        b = (
            b
            + """<taz edges="""
            + '"'
            + str(" ".join(x))
            + '" id="'
            + str(j[0])
            + '"'
            + """/>"""
        )  # +1 in the zone if for tomtom data

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
    while counter < 5:  # expected_trips - get_length_trip_files > 100:
        remove_source, remove_destination = get_disconnected_edges()
        new_taz_dict = filter_tazs(remove_source, remove_destination)
        write_tazs(new_taz_dict)
        generate_trips()
        counter += 1

    print("Done")
