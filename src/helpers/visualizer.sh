# Define the folder and scenario
export folder=free_nd_3600_1_10_0.8_0.2_0_0_sequential_naive_2a
export scenario=ua_aqt

# Plot custom speed measures
python2 ~/Documents/git/sumo/tools/visualization/plot_net_dump_custom_speeds.py \
  -v -n ../../$scenario/network.net.xml \
  --measures speed,speed \
  -i ../../$scenario/$folder/best_edge_data_3600,../../$scenario/$folder/edge_data_3600 \
  -o ../../$scenario/$folder/speed270%s.png \
  --max-width 1 \
  --min-width 1 \
  --max-color-value 60 \
  --dpi 300 \
  -s 5,5 \
  -b \
  --colormap RdYlGn

# Plot custom flow measures
python2 ~/Documents/git/sumo/tools/visualization/plot_net_dump_custom_flows.py \
  -v -n ../../$scenario/network.net.xml \
  --measures ,left \
  -i ../../$scenario/$folder/best_edge_data_3600,../../$scenario/$folder/best_edge_data_3600 \
  -o ../../$scenario/$folder/influx_out470%s.png \
  --max-width 5 \
  --min-width 0.625 \
  --max-color-value 500 \
  --min-width-value 10 \
  --max-width-value 500 \
  --dpi 600 \
  -s 5,5 \
  -b
