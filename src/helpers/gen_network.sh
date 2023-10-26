#!/bin/bash

# Activate the virtual environment
source ~/venv/bin/activate

# Define environment variables
export input_edge_csv=network.edg.csv
export input_taz_csv=raw_tazes.taz.csv
export output_taz_csv=new_tazes.csv
export seed=5
export folder_name=../../ua_aqt/gen #random_sims_90_$seed #munich_sims_d80_$seed
export demand_multiplier=1
export det_incidence_threshold=.999

# Change directory to $folder_name
cd $folder_name

# Print the current working directory
pwd

# Generate network data
# netgenerate -r --rand.iterations 500 --rand.max-distance 2000 --default.lanenumber 2 --rand.min-distance 500 --rand.neighbor-dist1 0 --rand.neighbor-dist2 5 --rand.neighbor-dist3 1 --rand.neighbor-dist4 4 --rand.neighbor-dist5 1 --rand.neighbor-dist6 0 --rand.min-angle 80 -o network.net.xml --seed $seed

# Generate TAZ (Traffic Analysis Zones) from the network
python $SUMO_HOME/tools/district/gridDistricts.py -n ../network.net.xml -o raw_tazes.taz.xml -w 50
python $SUMO_HOME/tools/xml/xml2csv.py raw_tazes.taz.xml

# Convert network data to plain format
netconvert --sumo-net-file ../network.net.xml --plain-output-prefix network --proj.plain-geo

# Convert XML files to CSV
python $SUMO_HOME/tools/xml/xml2csv.py network.edg.xml -x $SUMO_HOME/data/xsd/edges_file.xsd
python $SUMO_HOME/tools/xml/xml2csv.py ../network.net.xml -x $SUMO_HOME/data/xsd/net_file.xsd

# Process TAZ data
python ../../src/helpers/taz_processor.py

# Convert CSV to XML
python $SUMO_HOME/tools/xml/csv2xml.py -d , $folder_name/$output_taz_csv -x $SUMO_HOME/data/xsd/taz_file.xsd -o $folder_name/tazes.taz.xml
python $SUMO_HOME/tools/xml/xml2csv.py tazes.taz.xml

# Uncomment and add additional steps if needed
# mkdir -p $folder_name/demand
# python ../src/generate_od.py
# python ../src/generate_additional_synthetic.py
