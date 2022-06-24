# TODO: add resampling, add filtering, CWT(continuous wavelet transform)
# TODO: record mic

# wight (gram)
# nothing: 0
# lemon: 41
# kiwi: 123
# orange1: 192
# orange2: 227
import glob,os
import scipy.fftpack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from utils import calc_fft_specfreq
from resample import resample as linear_interpl




def pca_sensor_xyz_xy(df):
    pca = PCA(n_components=1)
    df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']] -= df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].mean()
    df['acc_xyz_pca'] = pca.fit_transform( df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].to_numpy())
    df['acc_xy_pca'] = pca.fit_transform( df[['acc_X_value', 'acc_Y_value']].to_numpy())
    return df


def pca_sensor(df):
    pca = PCA(n_components=1)
    df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']] -= df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].mean()
    df['acc_pca'] = pca.fit_transform( df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].to_numpy())
    return df


def max_fft_amp(y, T=1/200.0):
    xf, fft_amp = calc_fft(y, T)

    df = pd.DataFrame({'freq':xf, 'fft': fft_amp})
    # print(df)
    fft_max = df['fft'].loc[df['fft'].idxmax()]
    # print(fft_max)

    return xf, fft_max


def calc_fft(y, T):
    y = y[:int(len(y)/2)*2]
    N = len(y)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    fft_amp = 2.0/N * np.abs(yf[:N//2])

    # fig, ax = plt.subplots()
    # ax.plot(xf, fft_amp)
    # plt.show()

    return xf, fft_amp


def resample_df(df):
    df = linear_interpl(df, time_col_header='time_tick', sampling_rate=200)
    
    # df['time'] = pd.to_datetime(df['time_tick'])
    # print(df)
    # df = df.set_index('time')
    # df = resampling_pandas(df)
    return df


def fruit(path,delay,starttime):
    df = pd.read_csv(path)
    df = resample_df(df)
    df = df[(df['time_tick']>(starttime+delay+6500))&(df['time_tick']<(starttime+delay+7500))]
    # df = df[(df['time_tick']>(starttime+delay+6000))&(df['time_tick']<(starttime+delay+8000))]
    df = pca_sensor_xyz_xy(df)
    # print(df)
    t = df['time_tick'].values
    t = (t - t[0])/1000
    x = df['acc_X_value'].values
    y = df['acc_Y_value'].values
    z = df['acc_Z_value'].values
    xyz_pca = df['acc_xyz_pca'].values
    xy_pca = df['acc_xy_pca'].values
    return t, x, y, z, xyz_pca, xy_pca


def fruit_no_load(path,delay,starttime):
    df = pd.read_csv(path)
    df = resample_df(df)
    df = df[(df['time_tick']>(starttime+4000))&(df['time_tick']<(starttime+5000))]
    df = pca_sensor_xyz_xy(df)
    # print(df)
    t = df['time_tick'].values
    t = (t - t[0])/1000
    x = df['acc_X_value'].values
    y = df['acc_Y_value'].values
    z = df['acc_Z_value'].values
    xyz_pca = df['acc_xyz_pca'].values
    xy_pca = df['acc_xy_pca'].values
    return t, x, y, z, xyz_pca, xy_pca






def main(src):

    input_path = src
    paths = [input_path]

    weights_all = []
    intensity_load_all = []
    intensity_no_load_all = []

    for path in paths:
        print('path:', path)
        weights = []
        intensity_load = []
        intensity_no_load = []
        current_file = path
        param = path.split("/")[-1].split('_')
        """
        Param index: SAPPLE_1_125_1593191386730_45000_0.8889_2
        0: Fruit name
        1: ID
        2: weight
        3: start time
        4: cycle
        5: delay
        6: ratio
        7: repeat
        """

        """
        fruit() will return the data starting from 6.5 to 7.5 second
        """
        tf, xf, yf, zf, xyz_pcaf, xy_pcaf = fruit(current_file,3000,int(param[0]))
        _, fft_yf = calc_fft(yf, T=200)
        freq_arr, max_fft_yf = max_fft_amp(yf)
        """
        fruit_no_load() will return the data starting from 4 to 5 second
        """
        te, xe, ye, ze, xyz_pcae, xy_pcae = fruit_no_load(current_file,3000,int(param[0]))
        _, fft_ye = calc_fft(ye, T=200)
        freq_arr, max_fft_ye = max_fft_amp(ye)


        yf -= np.mean(yf)
        intense_f = np.mean(np.abs(yf))
        intensity_load.append(intense_f)

        ye -= np.mean(ye)
        intense_e = np.mean(np.abs(ye))
        intensity_no_load.append(intense_e)
    #standard net intensity: 2.7
    time = 1.7 / np.asarray(intensity_no_load)
    intensity_net = np.asarray(intensity_no_load) - np.asarray(intensity_load)
    return str(intensity_net*70/1.7*np.asarray(intensity_no_load))
    #return str(np.asarray(intensity_no_load))





