import numpy as np
import matplotlib.pyplot as plt
import bog_generator as bg
from optparse import OptionParser

if __name__ == "__main__":
    data_generator = bg.BogDataGenerator()

    # Specify the following options: filename. num_jobs, and the options for BlochSpinSimulator and NMRSimulator.
    optparser = OptionParser()

    optparser.add_option("-f", "--filename", dest="filename", default="data", help="The filename to which the data will be saved.")
    optparser.add_option("-j", "--num_jobs", dest="num_jobs", default=10, help="The number of jobs to be run in parallel.")

    optparser.add_option("-1", "--T1", dest="T1", default=100e-4, help="Example setting 1")
    optparser.add_option("-2", "--T2", dest="T2", default=50e-4,  help="Example setting 2.")

    (options, args) = optparser.parse_args()


    opt_map_arr = []
    for i in range(options.num_jobs):
        opt_map = {}

        # Grab settings from the options (here just a random example)
        opt_map["T1"] = np.abs(options.T1 * (1 + 0.1*np.random.randn()))
        opt_map["T2"] = np.abs(options.T2 * (1 + 0.1*np.random.randn()))
        opt_map["runtime"] = 1;
        
        opt_map_arr.append(opt_map)


    data_generator.save_arrays_to_hdf5(data_generator.parallel_process(opt_map_arr), options.filename)