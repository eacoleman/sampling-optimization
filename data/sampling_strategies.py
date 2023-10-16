from matplotlib import pyplot as plt
import pandas as pd

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

def urs_predict(n, df, timestep): 
    #want to only sample within bog borders within the farm 
    #choose n random points from data of only one timestep 
    #if not in farm val is -0, if border val is 0 
    #only take positive carbon values to sample from 
    samples = df[(df['carbon']>0) & (df['timestep'] == timestep)].sample(n = n, random_state = 9)
    prediction = samples['carbon'].sum() / n 
    return prediction

def analyze_accuracy(df, n, timestep): 
    bog_points = len(df[(df['carbon']>0) & (df['timestep'] == timestep)])
    y = df[df['timestep'] == timestep]['carbon'].sum() / bog_points
    ns = np.arange(1, n, 100)
    err = [np.abs(urs_predict(n, df, timestep) - y)/y for n in ns]

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.plot(ns, err)
    plt.xlabel("n")
    plt.ylabel("Prediction Error %")
    plt.title("URS Prediction Error")