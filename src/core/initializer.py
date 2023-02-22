#### This class is used to make ready the files needed for the simulation after 
#### we have the network and TAZs files ready. Idea is to streamline the scenario creation
#### by consolidating all the codes in the Jupyter Notebook
# some things that need to be considered are
# OD files
# a blank od file with columns o, d and t and same number of rows as above
# real counts
# Initial sumo run and trip output file
# Weight matrix for WSPSA
# create positive negative additionals
# modify the out.xml in these additionals to out_negative and out_positive
# preserve scenario files to avoid running them again and again


class Sumo_Inputs_Initializer:
	self.a = 1


