#!/bin/bash
# Download an OSM network for a bounding box and convert it into a SUMO network.
# Requires osmfilter and SUMO's netconvert on the PATH.
set -euo pipefail

RAW_NETWORK=demo_raw_network.osm
CLEAN_NETWORK=demo_clean_network.osm
FINAL_NETWORK=demo_network.net.xml
BBOX=47.6985415483953190,11.8014099151749789,47.9689446658581744,12.2510342616238823

python3 "$(dirname "$0")/download_network.py" --bbox "$BBOX" --output "$RAW_NETWORK"
osmfilter "$RAW_NETWORK" --keep="highway" --drop="highway=footway =service =steps =bridleway =construction =path =proposed =raceway =track =barrier =living_street  =raceway  =corridor" > "$CLEAN_NETWORK" # =unclassified =residential =cycleway =pedestrian
netconvert --osm-files "$CLEAN_NETWORK" --remove-edges.isolated --ramps.guess --roundabouts.guess --edges.join --join-lanes --tls.guess.joining --junctions.join --junctions.join-dist 50 --no-turnarounds.except-turnlane --geometry.avoid-overlap --ramps.no-split --geometry.remove --tls.discard-simple --tls.guess-signals --tls.join --no-internal-links -o "$FINAL_NETWORK"
