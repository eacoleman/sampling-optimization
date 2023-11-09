from matplotlib import pyplot as plt
import pandas as pd
import scipy
import numpy as np

n = 1000
timestep = 0

def data_to_df(data):
    #data is a timestep x width x height matrix
    #convert to pandas dataframe with cols ['timestep', 'coord', 'carbon']
    timesteps, width, height = np.shape(data)
    data = data.ravel()
    df = pd.DataFrame(data = data, columns = ['carbon'])
    df['index'] = df.index
    df['timestep'] = df.index // (width * height)
    df['x_coord'] = df['index'] % width
    df['y_coord'] = (df['index'] - df['timestep'] * width * height) // width
    return df

def urs_predict(samples, n, df):
    #want to only sample within bog borders within the farm
    #choose n random points from data of only one timestep
    #if not in farm val is -0, if border val is 0
    #only take positive carbon values to sample from
    prediction = samples['carbon'].sum() / n
    return prediction

def get_urs_samples(df, n, timestep): 
    samples = df[(df['carbon']>0) & (df['timestep'] == timestep)].sample(n = n, random_state = 9)
    return samples

def get_interval_samples(df, n, timestep, total_points): 
    #Translate coordinate system to actual world distances for now 
    #Assume 1:1 mapping take samples where timestep = timestep
    #get every interval index
    #total_points // total_samples = interval to sample at 
    interval = total_points // n
    sample = df[(df['index']%interval == 0) & (df['timestep'] == timestep)].head(n)
    x1s = sample['x_coord'].tolist()
    x2s = sample['y_coord'].tolist()
    return x1s, y1s

def analyze_accuracy(df, n, timestep):
    bog_points = len(df[(df['carbon']>0) & (df['timestep'] == timestep)])
    y = df[df['timestep'] == timestep]['carbon'].sum() / bog_points
    ns = np.arange(1, n, 100)
    samples = get_urs_samples(df, n, timestep)
    err = [np.abs(urs_predict(samples, n, df) - y)/y for n in ns]

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(ns, err)
    plt.xlabel("n")
    plt.ylabel("Prediction Error %")
    plt.title("URS Prediction Error")

def get_distribution(x1s, y1s): 
    return np.ndarray.flatten(np.sqrt(np.subtract.outer(x1s,x1s)**2 + np.subtract.outer(y1s,y1s)**2))

def in_bounds(data, x = 0, y = 0):
    if x < 0 or y < 0 or x >= data.shape[1] or y >= data.shape[1]: 
        return False
    return True

#shift from distribution a to b via earth mover's distance
def shift_distributions(sample_a, distribution_a, distribution_b, iterations, n, df, data):

    bog_points = len(df[(df['carbon']>0) & (df['timestep'] == timestep)])
    y = df[df['timestep'] == timestep]['carbon'].sum() / bog_points

    pred_err = [np.abs(urs_predict(sample_a, n, df) - y)/y]

    min_distance = scipy.stats.wasserstein_distance(distribution_a, distribution_b)
    while iterations > 0: 
        #make small variations to the coords sampled by sample_a
        #choose random number between 1 and 50 in range of farm bounds
        #for all coords and create new list of things 
        random_permutations = np.random.randint(-50, 50, n)
        xs = sample_a['x_coord'].tolist()
        ys = sample_a['y_coord'].tolist()
        delta_xs = np.add(xs, random_permutations, where = np.where(lambda x, r: in_bounds(data, x = x + r)))
        delta_ys = np.add(ys, random_permutations, where = np.where(lambda x, r: in_bounds(data, y = y+ r)))
        distribution_a_delta = get_distribution(delta_xs, delta_ys)
        distance = scipy.stats.wasserstein_distance(distribution_a_delta, distribution_b)
        #accept changes
        if distance < min_distance: 
            sample_a= df[(df['x_coord'].is_in(delta_xs)) & (df['y_coord'].is_in(delta_ys)) & (df['timestep'] == timestep)]
        #calculate new accuracy and append
        pred_err.append(np.abs(urs_predict(sample_a, n, df) - y)/y)
        iterations -= 1

    return sample_a, pred_err

def plot_wasserstein_err(err, iterations):
    #Plot error over time with earth mover's 
    ns = np.arange(0, iterations + 1, 1)
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(ns, err)
    plt.xlabel("Iterations")
    plt.ylabel("Prediction Error %")
    plt.title("Prediction Error over Earth Mover's Iterations")

def urs_interval_analysis(data):
    df = data_to_df(data)

    total_points = np.shape(data)[1] * np.shape(data)[2]
    i_xs, i_ys = get_interval_samples(df, n, 0, total_points)

    dist_interval = get_distribution(i_xs, i_ys)
    urs_samples = get_urs_samples(df, n, timestep)

    x1s = urs_samples['x_coord'].tolist()
    y1s = urs_samples['y_coord'].tolist()
    #x1s and y1s are the coordinates of the samples 

    dist_urs = get_distribution(x1s, y1s)

    plt.hist(dist_urs, bins=100, density=True, label = "Uniform Random Sampling n = " + str(n))
    plt.hist(dist_interval, bins=100, density=True, label= "Evenly Spaced Sampling n = " + str(n), alpha=0.6)
    plt.xlabel("Distance between samples",fontsize=16)
    plt.ylabel("Normalized probability density",fontsize=14)
    plt.legend()
    plt.show()

    iterations = 100
    shifted_urs, err = shift_distributions(urs_samples, dist_urs, dist_interval, iterations, n, df)

    #Plot final distribution with smaller earth mover's distance
    shifted_x1s = shifted_urs['x_coord'].tolist()
    shifted_y1s = shifted_urs['y_coord'].tolist()
    shifted_urs_dist = get_distribution(shifted_x1s, shifted_y1s)
    plt.hist(shifted_urs_dist, bins=100, density=True, label = "Shifted Uniform Random Sampling n = " + str(n))
    plt.hist(dist_interval, bins=100, density=True, label= "Evenly Spaced Sampling n = " + str(n), alpha = .6)
    plt.xlabel("Distance between samples",fontsize=16)
    plt.ylabel("Normalized probability density",fontsize=14)
    plt.legend()
    plt.show()