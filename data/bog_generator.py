import numpy as np
import multiprocessing
import h5py
import torch.utils.data as data

from tqdm.contrib.concurrent import process_map  # or thread_map


"""
    ### BogDataGenerator

    Description:

        A handler class which generates a large quantity of 
        bog carbon stock data from a set of parameter ranges
        and saves them to an HDF5 file.

"""
class BogDataGenerator :
    def __init__(self) :
        pass


    ## get_bog_data
    #
    # Description:
    #   A function which takes in a dictionary of options and returns an array of
    #   bog data simulated using the options. The output should be a 3D array of
    #   size (number of time steps, plot size in pixels, plot size in pixels).
    def get_bog_data(self, opt_map):

        # TODO: Fill in with sticky Markov model output here.
        return np.random.randn(100)


    ## process_job
    # 
    # Description: 
    #   A function which takes in a dictionary of options and returns a tuple of
    #   (bog_data, opt_map) where bog_data is the simulated data and opt_map is the
    #   options dictionary.
    def process_job(index, opt_map):
        
        # Define the required keys in the opt_map
        # T1 and T2 are example 1 and example 2 of inputs you could pass to the simulator
        required_keys = ["T1", "T2", "runtime"]

        # Check if all the required keys are present in opt_map
        if not all(key in opt_map for key in required_keys):
            missing_keys = [key for key in required_keys if key not in opt_map]
            raise ValueError(f"Missing options in opt_map: {', '.join(missing_keys)}")

        # Use default settings if options are not otherwise specified
        example_unspecified_setting   = opt_map.get("example_unspecified_setting"  , 2e-5)

        # Use opt_map to get the specified sample data
        bog_data = self.get_bog_data(opt_map)

        # Return the measured magnetization (for input)
        # and the sim options
        return (bog_data, opt_map)


    ## parallel_process
    # 
    # Description:
    #   A function which takes in an array of opt_maps and processes them in parallel.
    #   Returns an array of tuples of (bog_data, opt_map) where bog_data is the simulated
    #   data and opt_map is the options dictionary.
    def parallel_process(self, opt_map_arr):
        
        print("Processing",len(opt_map_arr),"jobs in parallel")
        print("\t - Using",multiprocessing.cpu_count(),"workers")

        # Create a multiprocessing pool with the number of processes equal to the number of CPU cores
        job_outputs = process_map(self.process_job, opt_map_arr, max_workers=multiprocessing.cpu_count())

        print("Finished parallel processing",len(job_outputs),"jobs.")

        return job_outputs


    ## save_arrays_to_hdf5
    #
    # Description:
    #   A function which takes in an array of tuples of (bog_data, opt_map) and saves them
    #   to an HDF5 file.
    def save_arrays_to_hdf5(self, job_outputs, filename):
        if not isinstance(job_outputs, list):
            job_outputs = [job_outputs]

        print("Saving jobs to file",filename+".h5")

        if not h5py.is_hdf5(filename+".h5"):
            # If the file doesn't exist, create a new one and save the arrays
            with h5py.File(filename+".h5", 'w') as hfm:

                for i, (bog_data, opt_map) in enumerate(job_outputs):
                    dset = hfm.create_dataset(f'bog_{i}', data=bog_data)
                    dset.attrs["T1"] = opt_map["T1"]
                    dset.attrs["T2"] = opt_map["T2"]
                    dset.attrs["runtime"] = opt_map["runtime"]
                    
        else:
            # If the file already exists, merge the existing datasets with the new arrays
            print("\t - File already exists, merging datasets.")

            with h5py.File(filename+".h5", 'r+') as hfm:
                num_existing_arrays = len(hfm.keys())
                
                print("\t - Found",num_existing_arrays,"existing experiments in dataset.")

                for i, (bog_data, opt_map) in enumerate(job_outputs):
                    dset = hfm.create_dataset(f'bog_{num_existing_arrays + i}', data=bog_data)
                    dset.attrs["T1"] = opt_map["T1"]
                    dset.attrs["T2"] = opt_map["T2"]
                    dset.attrs["runtime"] = opt_map["runtime"]