import pandas as pd
from collections import Counter
import os


class FilterTAZByEdges:
    """
    A class for filtering TAZ data based on specified edges.
    """

    def __init__(self, input_edge_csv, input_taz_csv, output_taz_csv):
        """
        Initialize the FilterTAZByEdges object.

        Parameters:
        - input_edge_csv (str): Path to the input edge CSV file.
        - input_taz_csv (str): Path to the input TAZ CSV file.
        - output_taz_csv (str): Path to the output TAZ CSV file.
        """
        self.input_edge_csv = input_edge_csv
        self.input_taz_csv = input_taz_csv
        self.output_taz_csv = output_taz_csv

    def filter_taz_by_edges(self):
        """
        Filter TAZ data based on specified edges and save the result to a CSV file.
        """
        df_network = pd.read_csv(self.input_edge_csv, sep=";")
        list_edges = []

        for node, degree in Counter(df_network.edge_from).most_common():
            if degree == 1:
                list_edges.append(
                    df_network[df_network.edge_from == node]["edge_id"].values[0]
                )

        df_taz = pd.read_csv(self.input_taz_csv, sep=";")
        taz_id_keep = []

        for taz_id, taz_edges in zip(df_taz.taz_id, df_taz.taz_edges):
            edges = [i for i in taz_edges.split(" ")]
            for edge in edges:
                if edge in list_edges:
                    taz_id_keep.append(taz_id)
                    continue

        df_taz_new = df_taz[df_taz.taz_id.isin(taz_id_keep)]
        df_taz_new.to_csv(self.output_taz_csv, index=None)


if __name__ == "__main__":
    # Example usage:
    input_edge_csv = os.environ.get("input_edge_csv")
    input_taz_csv = os.environ.get("input_taz_csv")
    output_taz_csv = os.environ.get("output_taz_csv")

    filter_tool = FilterTAZByEdges(input_edge_csv, input_taz_csv, output_taz_csv)
    filter_tool.filter_taz_by_edges()
