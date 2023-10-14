import numpy as np
import multiprocessing
import h5py
import torch.utils.data as data
import matplotlib as plt

from tqdm.contrib.concurrent import process_map  # or thread_map


"""
    ### BogDataGenerator

    Description:

        A handler class which generates a large quantity of 
        bog carbon stock data from a set of parameter ranges
        and saves them to an HDF5 file.

"""
class BogDataGenerator :
    def __init__(self, farm) :
        # Should be instance of farm class to make sure masking works on borders
        self.farm = farm
        self.num_steps = 1000
        self.p_stickiness = 1.0 
    
    def peatland_generation(self):
        """
        PREVIOUS GENERATION BASED ON PLOTTING RANDOM GAUSSIANS 
        FEEL FREE TO REMOVE JUST ADDING TO GITHUB FOR HISTORY 
        plots = farm.plots
        samples = 10
        #instead need to randomize mean and standard dev to generate gaussian over it
        #the point where the distribution is centered aka mean

        rng = np.random.default_rng()
        coords = list(plots.keys())
        #TODO: scale variance based on size of the farm
        min_var = 500
        max_var = 2000
        peat_locs = rng.choice(coords, size = samples, replace=False)
        variance_x = np.random.randint(min_var, max_var, size = samples)
        variance_y = np.random.randint(10, 100, size = samples)
        cov = np.random.randint(10, 100, size = samples)
        variances = []

        #map these to the varying peat accumulation heights
        max_density = 1/np.sqrt(2*np.pi*min_var**2)
        min_density = 1/np.sqrt(2*np.pi*max_var**2)

        for ii in range(samples):
        cov_m = [[variance_x[ii], 0], [0, variance_x[ii]]]
        variances.append(cov_m)

        #so just check if nan for visualization
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(projection='3d')
        for ii in range(samples):
        x, y, z = np_bivariate_normal_pdf(peat_locs[ii], variances[ii], farm)
        #modify specific parts of the elevation mesh based on coords of x, y
        start_x, end_x = int(x[0][0]), int(x[0][-1])
        start_y, end_y = int(y[0][0]), int(y[-1][0])

        x_off, y_off = farm.bounding_box[0][0], farm.bounding_box[0][1]

        plt_plot_bivariate_normal_pdf(x, y, z, ax)

        farm.elevation_mesh[start_x - x_off:end_x - x_off + 1, start_y - y_off:end_y - y_off + 1] = z.T

        plt.show()

        #TODO: scale farm elevation before plotting
        scaled_elevation_mesh = (farm.elevation_mesh) / (max_density) * 200
        farm.elevation_mesh = scaled_elevation_mesh

        #apply mask np array only plot parts of surface where z not nan (ie: points in farm)
        masked_elev = np.ma.masked_where(np.isnan(farm.coord_mask), scaled_elevation_mesh)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(farm.X_mesh, farm.Y_mesh, masked_elev.T, cmap=cm.coolwarm, linewidth=0, antialiased=True)

        #make the z axis less crazy please
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Peat Height');
        ax.set_zlim3d(0, 200)
        plt.show()
        """
        #instead generate based on sticky markov model for more peatlike looking things 
        plots = self.farm.plots 
        from scipy.ndimage import gaussian_filter 

        box = self.farm.bounding_box 
        #make grid of correct size 
        coarse_grained_factor = 4
        grid_size = (self.farm.bounding_box[3][0] - self.farm.bounding_box[0][0] + 1, self.farm.bounding_box[3][1] - self.farm.bounding_box[0][1] + 1)

        resulting_grid = generate_sticky_markov_model(grid_size, self.num_steps, self.p_stickiness)

        # Coarse-grain by averaging over blocks
        coarse_grained_grid = resulting_grid[:(grid_size[0] // coarse_grained_factor * coarse_grained_factor),:(grid_size[1] // coarse_grained_factor * coarse_grained_factor)].reshape(grid_size[0] // coarse_grained_factor,
                                                coarse_grained_factor,
                                                grid_size[1] // coarse_grained_factor,
                                                coarse_grained_factor).mean(axis=(1, 3))
        # Upsample using nearest neighbor interpolation
        upsampled_grid = np.kron(coarse_grained_grid, np.ones((coarse_grained_factor, coarse_grained_factor)))

        #Pad upsampled grid such that masking over farm boundary works 
        x_pad_width = grid_size[0] - np.shape(upsampled_grid)[0] + 1
        y_pad_width = grid_size[1] - np.shape(upsampled_grid)[1] + 1
        upsampled_grid = np.pad(upsampled_grid, ((x_pad_width, 0), (y_pad_width, 0)), mode = "edge")

        # Smooth the upsampled image
        smoothed_grid = gaussian_filter(upsampled_grid, sigma=4)

        # Find the boundary
        # Currently boundary is not based on "farm" image boundaries
        # TODO: Could potentially change this
        grid_boundary = np.abs(np.diff(smoothed_grid > 0,axis=0)[:,:-1]) + np.abs(np.diff(smoothed_grid > 0,axis=1))[:-1,:]
        grid_boundary = (grid_boundary > 0).astype(int)

        grid_boundary[0,:]  = smoothed_grid[0,:-1]  > 0
        grid_boundary[:,0]  = smoothed_grid[:-1,0]  > 0
        grid_boundary[-1,:] = smoothed_grid[-1,:-1] > 0
        grid_boundary[:,-1] = smoothed_grid[:-1,-1] > 0

        # Plot the results
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(resulting_grid, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Sticky Markov Model')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 2, 2)
        plt.imshow(smoothed_grid > 0, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Smoothed and Upsampled')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(2, 2, 3)
        plt.imshow(grid_boundary, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Boundary')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.tight_layout()
        plt.show()

        return grid_boundary, smoothed_grid

    def simulate_peat_accrual_finite_difference(self, X, Y, grid_boundary, smoothed_grid): 
        fig, ax = plt.subplots(figsize=(20, 5))

        # Parameters
        nx, ny = grid_boundary.shape
        dx = 1.0 / (nx - 1)
        dy = 1.0 / (ny - 1)

        u = smoothed_grid[:-1, :-1]

        # Initialize p with smoothed_grid within the boundary
        p = (u > 0) * (u + 0.01)
        p[(grid_boundary > 0)] = 0.0

        # Plot the solution within the region defined by the boundary
        # mask to only include regions inside the farm 
        masked_p = np.ma.masked_where(np.isnan(self.farm.coord_mask), p)
        print(np.shape(masked_p))

        data = np.array([masked_p])
        print(np.shape(data))

        plt.subplot(1, 4, 1)
        IS = plt.contourf(X[:nx,:ny], Y[:nx,:ny], masked_p, cmap='viridis')
        plt.colorbar()
        plt.contour(IS, levels=[0], colors='r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Initial guess')


        # Source term
        f = - 1000 * np.ones_like(grid_boundary)

        # Iterative solver (Gauss-Seidel) with Dirichlet boundary conditions
        max_iter = 300
        peat_accrual = np.zeros(max_iter+1)
        residuals = np.zeros(max_iter)

        for iteration in range(max_iter):
            peat_accrual[iteration] = np.sum(p)

            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    if grid_boundary[i,j] == 0 and u[i,j] > 0:  # Only update non-boundary interior points
                        p[i, j] = 0.25 * (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - dx * dy * f[i, j])

            masked_p = np.ma.masked_where(np.isnan(self.farm.coord_mask), p)
            next_data = np.array([masked_p])
            data = np.vstack((data, next_data))

            # Calculate the Laplacian of u using finite differences and compare to the source term.
            # If the residual is 0 everwhere within the boundary, then the solution satisfies Poisson's equation.
            laplacian_p = (p[1:-1, :-2] + p[:-2, 1:-1] - 4 * p[1:-1, 1:-1] + p[1:-1, 2:] + p[2:, 1:-1]) / (dx ** 2)
            residual = (laplacian_p - f[1:-1, 1:-1]) * ((u > 0)*(grid_boundary == 0))[1:-1, 1:-1]
            residuals[iteration] = np.sum(np.abs(residual))


        peat_accrual[-1] = np.sum(p)

        plt.subplot(1, 4, 2)
        CS = plt.contourf(X[:nx,:ny], Y[:nx,:ny], masked_p, cmap='viridis')
        plt.colorbar()
        CS2 = plt.contour(CS, levels=[0], colors='r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('After solving Poisson\'s Equation')

        plt.subplot(1, 4, 3)
        plt.plot(peat_accrual)
        plt.xlabel("Iteration")
        plt.ylabel('Net Peat in Bog')
        plt.title('Peat Accrual over Iterations')

        plt.subplot(1, 4, 4)
        plt.plot(residuals)
        plt.xlabel("Iteration")
        plt.ylabel('Net error in Poisson\'s Equation')
        plt.yscale('log')
        plt.title('Convergence to Poissonian behavior over Iterations')

        plt.tight_layout()
        plt.show()
        
        #Should be in shape (timesteps + 1, x, y)
        print(np.shape(data))
        return data
        
    ## get_bog_data
    #
    # Description:
    #   A function which takes in a dictionary of options and returns an array of
    #   bog data simulated using the options. The output should be a 3D array of
    #   size (number of time steps, plot size in pixels, plot size in pixels).
    def get_bog_data(self, opt_map):
        grid_boundary, smoothed_grid  = self.peatland_generation()
        X = self.farm.X_mesh.T
        Y = self.farm.Y_mesh.T
        data = self.simulate_peat_accrual_finite_difference(X, Y, grid_boundary, smoothed_grid)
        return data
    
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

def generate_sticky_markov_model(grid_size, num_steps, p_stickiness):
        # Initialize the grid with random values
        grid = np.random.choice([-1, 0, 1], size=grid_size)

        for step in range(num_steps):
            new_grid = np.copy(grid)
            for i in range(1, grid_size[0] - 1):
                for j in range(1, grid_size[1] - 1):
                    neighbors = [
                        grid[i-1, j-1], grid[i-1, j], grid[i-1, j+1],
                        grid[i, j-1], grid[i, j+1],
                        grid[i+1, j-1], grid[i+1, j], grid[i+1, j+1]
                    ]
                    if grid[i, j] == 1:
                        if np.random.random() > p_stickiness:
                            new_grid[i, j] = np.random.choice([-1, 0])
                    elif grid[i, j] == -1:
                        if np.random.random() > p_stickiness:
                            new_grid[i, j] = np.random.choice([1, 0])
                    else:
                        if np.random.random() > p_stickiness:
                            new_grid[i, j] = np.random.choice([1, -1])

            grid = new_grid

        return grid