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


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# def resampling_pandas(df, sampling_freq=200, higher_freq=1000):
#     ''' Resample unevenly spaced timeseries data linearly by 
#     first upsampling to a high frequency (short_rate) 
#     then downsampling to the desired rate.
#     Parameters
#     ----------
#         df:               dataFrame
#         sampling_freq:    sampling frequency
#         max_gap_sec:      if gap larger than this, interpolation will be avoided
    
#     Return
#     ------
#         result:           dataFrame
        
#     Note: You will need these 3 lines before resampling_pandas() function
#     ---------------------------------------------------------------------
#         # df['date'] = pd.to_datetime(df['Time'],unit='ms')
#         # df = df.set_index(['date'])
#         # df.index = df.index.tz_localize('UTC').tz_convert(settings.TIMEZONE)
#     '''
    
#     # find where we have gap larger than max_gap_sec
#     # print(df.index)
#     # diff = np.diff(df.index)

#     # print(diff)
#     # idx = np.where(np.greater(np.diff(df.index), 1000))[0]
#     # start = df.index[idx].tolist()
#     # stop = df.index[idx + 1].tolist()
#     # big_gaps = list(zip(start, stop))

#     # upsample to higher frequency
#     df = df.resample('{}ms'.format(1000/higher_freq)).mean().interpolate()

#     # downsample to desired frequency
#     df = df.resample('{}ms'.format(1000/sampling_freq)).ffill()

#     # remove data inside the gaps
#     # for start, stop in big_gaps:
#         # df[start:stop] = None
#     df.dropna(inplace=True)

#     return df


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


def plt_raw_pca(t, acc):
    fig, ax1 = plt.subplots(nrows=1)
    ax1.plot(t, acc)
    # ax1.set_ylim(-0.2, 0.2)
    plt.show()


def plt_raw_xyz(t, x, y, z):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.plot(t, x)
    ax1.set_ylim(-0.2, 0.2)
    ax2.plot(t, y)
    ax2.set_ylim(-0.2, 0.2)
    ax3.plot(t, z)
    ax3.set_ylim(9.6, 10)
    plt.show()


def plt_raw_spec(t, x, y, z, NFFT=128, Fs=100, noverlap=90):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)
    ax1.plot(t, x)
    ax1.set_ylim(-0.2, 0.2)
    ax2.plot(t, y)
    ax2.set_ylim(-0.2, 0.2)
    ax3.plot(t, z)
    ax3.set_ylim(9.6, 10)
    Pxx, freqs, bins, im = ax4.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    Pxx, freqs, bins, im = ax5.specgram(y, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    Pxx, freqs, bins, im = ax6.specgram(z, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    plt.show()


def nothing_pca():
    df = pd.read_csv('../data/noobject-acc.csv')
    df = pca_sensor(df)
    df = df[(df['time_tick']<32) & (df['time_tick']>12)]
    t = df['time_tick'].values
    acc = df['acc_pca'].values
    return t, acc


def lemon_pca():
    df = pd.read_csv('../data/lemon41gbound-acc.csv')
    df = pca_sensor(df)
    df = df[(df['time_tick']<90) & (df['time_tick']>70)]
    t = df['time_tick'].values
    acc = df['acc_pca'].values
    return t, acc


def kiwi_pca():
    df = pd.read_csv('../data/kiwi123gbound-acc.csv')
    df = pca_sensor(df)
    df = df[(df['time_tick']<35) & (df['time_tick']>15)]
    t = df['time_tick'].values
    acc = df['acc_pca'].values
    return t, acc


def orange1_pca():
    df = pd.read_csv('../data/orange192gbound-acc.csv')
    df = pca_sensor(df)
    df = df[(df['time_tick']<95) & (df['time_tick']>75)]
    t = df['time_tick'].values
    acc = df['acc_pca'].values
    return t, acc


def orange2_pca():
    df = pd.read_csv('../data/orange227gbound-acc.csv')
    df = pca_sensor(df)
    df = df[(df['time_tick']<35) & (df['time_tick']>15)]
    t = df['time_tick'].values
    acc = df['acc_pca'].values
    return t, acc


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


def compare_raw_intensity1():
    t, x, y, z = nothing()
    l0 = y
    l0 = l0 - np.mean(l0)

    t, x, y, z = fruit('../data/168_1593108684617_45000_0.8889_2.csv',45000*(1-0.8889),1593108684617)
    l1 = y
    l1 = l1 - np.mean(l1)

    t, x, y, z = fruit('../data/173_1593109144033_45000_0.8889_2.csv',45000*(1-0.8889),1593109144033)
    l2 = y
    l2 = l2 - np.mean(l2)
    
    t, x, y, z = fruit('../data/183_1593109002630_45000_0.8889_2.csv',45000*(1-0.8889),1593109002630)
    l3 = y
    l3 = l3 - np.mean(l3)
    
    t, x, y, z = fruit('../data/204_1593108937441_45000_0.8889_2.csv',45000*(1-0.8889),1593108937441)
    l4 = y
    l4 = l4 - np.mean(l4)

    L = min(len(l0), min(len(l1), min(len(l2), min(len(l3), len(l4)))))
    t = t[:L]
    l0 = l0[:L]
    l1 = l1[:L]
    l2 = l2[:L]
    l3 = l3[:L]
    l4 = l4[:L]

    orig_I = np.mean(np.abs(l0))

    # Trial 2, as shown in slide    
    print("{0:.6f}".format((np.mean(np.abs(l0)))*gram['l0']))
    print("{0:.6f}".format((np.mean(np.abs(l1)))*gram['l1']))
    print("{0:.6f}".format((np.mean(np.abs(l2)))*gram['l2']))
    print("{0:.6f}".format((np.mean(np.abs(l3)))*gram['l3']))
    print("{0:.6f}".format((np.mean(np.abs(l4)))*gram['l4']))

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5)
    ax0.plot(t, l0)
    ax0.set_ylim(-.15, .15)
    ax1.plot(t, l1)
    ax1.set_ylim(-.15, .15)
    ax2.plot(t, l2)
    ax2.set_ylim(-.15, .15)
    ax3.plot(t, l3)
    ax3.set_ylim(-.15, .15)
    ax4.plot(t, l4)
    ax4.set_ylim(-.15, .15)
    plt.show()


def compare_pca_raw():
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
    t, acc = nothing_pca()
    acc = acc - np.mean(acc)
    ax1.plot(t, acc)
    ax1.set_ylim(-0.1, 0.1)

    t, acc = lemon_pca()
    acc = acc - np.mean(acc)
    ax2.plot(t, acc)
    ax2.set_ylim(-0.1, 0.1)
    
    t, acc = kiwi_pca()
    acc = acc - np.mean(acc)
    ax3.plot(t, acc)
    ax3.set_ylim(-0.1, 0.1)
    
    t, acc = orange1_pca()
    acc = acc - np.mean(acc)
    ax4.plot(t, acc)
    ax4.set_ylim(-0.1, 0.1)
    
    t, acc = orange2_pca()
    acc = acc - np.mean(acc)
    ax5.plot(t, acc)
    ax5.set_ylim(-0.1, 0.1)

    plt.show()


def compare_pca_spec(NFFT=128, Fs=100, noverlap=90):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
    t, acc = nothing_pca()
    acc = acc - np.mean(acc)
    Pxx, freqs, bins, im = ax1.specgram(acc, NFFT=NFFT, Fs=Fs, noverlap=noverlap)

    t, acc = lemon_pca()
    acc = acc - np.mean(acc)
    Pxx, freqs, bins, im = ax2.specgram(acc, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    
    t, acc = kiwi_pca()
    acc = acc - np.mean(acc)
    Pxx, freqs, bins, im = ax3.specgram(acc, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    
    t, acc = orange1_pca()
    acc = acc - np.mean(acc)
    Pxx, freqs, bins, im = ax4.specgram(acc, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    
    t, acc = orange2_pca()
    acc = acc - np.mean(acc)
    Pxx, freqs, bins, im = ax5.specgram(acc, NFFT=NFFT, Fs=Fs, noverlap=noverlap)

    plt.show()



def compare_pca_fft(T=1/100.0):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)

    t, acc = nothing_pca()
    acc = acc[:int(len(acc)/2)]
    acc = acc - np.mean(acc)
    xf, yf = calc_fft(acc)
    ax1.plot(xf, 2.0/len(acc) * np.abs(yf[:len(acc)//2]))
    ax1.set_ylim(0, 0.01)

    t, acc = lemon_pca()
    acc = acc[:int(len(acc)/2)]
    acc = acc - np.mean(acc)
    xf, yf = calc_fft(acc)
    ax2.plot(xf, 2.0/len(acc) * np.abs(yf[:len(acc)//2]))
    ax2.set_ylim(0, 0.01)

    t, acc = kiwi_pca()
    acc = acc[:int(len(acc)/2)]
    acc = acc - np.mean(acc)
    xf, yf = calc_fft(acc)
    ax3.plot(xf, 2.0/len(acc) * np.abs(yf[:len(acc)//2]))
    ax3.set_ylim(0, 0.01)

    t, acc = orange1_pca()
    acc = acc[:int(len(acc)/2)]
    acc = acc - np.mean(acc)
    xf, yf = calc_fft(acc)
    ax4.plot(xf, 2.0/len(acc) * np.abs(yf[:len(acc)//2]))
    ax4.set_ylim(0, 0.01)

    t, acc = orange2_pca()
    acc = acc[:int(len(acc)/2)]
    acc = acc - np.mean(acc)
    xf, yf = calc_fft(acc)
    ax5.plot(xf, 2.0/len(acc) * np.abs(yf[:len(acc)//2]))
    ax5.set_ylim(0, 0.01)

    plt.show()


def specgram_eg():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    dt = 0.0005
    t = np.arange(0.0, 20.0, dt)
    s1 = np.sin(2 * np.pi * 100 * t)
    s2 = 2 * np.sin(2 * np.pi * 400 * t)

    # create a transient "chirp"
    s2[t <= 10] = s2[12 <= t] = 0

    # add some noise into the mix
    nse = 0.01 * np.random.random(size=len(t))

    x = s1 + s2 + nse  # the signal
    NFFT = 1024  # the length of the windowing segments
    Fs = int(1.0 / dt)  # the sampling frequency

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(t, x)
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot
    plt.show()


if __name__ == '__main__':

    # ===============================================================
    apple_path = '../Data/7.2_nexus2_rotate/apple'
    tableware_path = '../Data/7.2_nexus2_rotate/tools'
    onion_path = '../Data/7.4_nexus2_rotate/onion'
    pepper_path = '../Data/7.4_nexus2_rotate/pepper'
    potato_path = '../data/7.5_nexus2_rotate/potato'
    ALL_path = '../Data/ALL'

    #validation = 'LOOCV'
    validation = '6-fold'
    # ===============================================================
    # paths = [apple_path, onion_path, pepper_path]
    paths = [potato_path,apple_path]

    weights_all = []
    intensity_load_all = []
    intensity_no_load_all = []

    for path in paths:
    # path = potato_path # apple_path # potato_path # onion_path #  # tableware_path # pepper_path #  # #

        if validation == '6-fold':

            if path == apple_path:
                n_splits = 6
            elif path == potato_path:
                n_splits = 6
            elif path == tableware_path:
                n_splits = 6
            elif path == pepper_path:
                n_splits = 6
            elif path == onion_path:
                n_splits = 6
            else:
                print('error!')
                exit()

        elif validation == 'LOOCV':

            # TODO: can use function to get the info from counting files

            if path == apple_path:
                n_splits = 24
            elif path == tableware_path:
                n_splits = 6
            elif path == pepper_path:
                n_splits = 6
            elif path == onion_path:
                n_splits = 16
            else:
                print('error!')
                exit()

        os.chdir(path)

        print('path:', path)
        print('n_splits:', n_splits)

        weights = []

        intensity_load = []
        intensity_no_load = []
        for counter, current_file in enumerate(glob.glob("*.csv")):
            param = current_file.split('_')
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
            tf, xf, yf, zf, xyz_pcaf, xy_pcaf = fruit(current_file,int(param[4]),int(param[3]))
            # print('with-load:')
            _, fft_yf = calc_fft(yf, T=200)
            freq_arr, max_fft_yf = max_fft_amp(yf)
            # print(max_fft_yf)
            """
            fruit_no_load() will return the data starting from 4 to 5 second
            """
            te, xe, ye, ze, xyz_pcae, xy_pcae = fruit_no_load(current_file,int(param[4]),int(param[3]))
            # print('zero-load:')
            _, fft_ye = calc_fft(ye, T=200)
            freq_arr, max_fft_ye = max_fft_amp(ye)
            # print(max_fft_ye)

            weights.append(int(param[2]))
            #intense = (np.mean(np.abs(y)))*int(param[2])
            

            # # ========================================
            # # FFT Dominant component:  MAE = 21.8
            # # ========================================
            # intensity_load.append(max_fft_yf)
            # intensity_no_load.append(max_fft_ye)

            # # ========================================
            # # FFT all component (0-100Hz):  MAE = 21.5
            # # ========================================
            # intensity_load.append(np.sum(fft_yf))
            # intensity_no_load.append(np.sum(fft_ye))

            # # ======================================
            # # x-axis: MAE = 15.7
            # # ========================================
            # xf -= np.mean(xf)
            # intense = np.mean(np.abs(xf))
            # print("{} {} {} {}:{:.6f}\n".format(param[0],param[1],param[2], param[3],intense))
            # intensity_load.append(intense)

            # xe -= np.mean(xe)
            # intense = np.mean(np.abs(xe))
            # intensity_no_load.append(intense)

            # ======================================
            # y-axis: MAE = 12.7
            # ========================================
            yf -= np.mean(yf)
            intense_f = np.mean(np.abs(yf))
            # print("{} {} {} {}:{:.6f}\n".format(param[0],param[1],param[2], param[3],intense))
            intensity_load.append(intense_f)

            ye -= np.mean(ye)
            intense_e = np.mean(np.abs(ye))
            intensity_no_load.append(intense_e)

            # # ======================================
            # # z-axis: MAE = 24.2
            # # ========================================
            # zf -= np.mean(zf)
            # intense = np.mean(np.abs(zf))
            if intense_e < intense_f:
                print("{} {} {} {}:{:.6f}\n".format(param[0],param[1],param[2], param[3],intense_e))
                # exit()
            # intensity_load.append(intense)

            # ze -= np.mean(ze)
            # intense = np.mean(np.abs(ze))
            # intensity_no_load.append(intense)

            # # ========================================
            # # XY PCA:  MAE = 15.1
            # # ========================================
            # xy_pcaf -= np.mean(xy_pcaf)
            # intense = np.mean(np.abs(xy_pcaf))
            # print("{} {} {} {}:{:.6f}".format(param[0],param[1],param[2], param[3],intense))
            # intensity_load.append(intense)

            # xy_pcae -= np.mean(xy_pcae)
            # intense = np.mean(np.abs(xy_pcae))
            # intensity_no_load.append(intense)

            # # ========================================
            # # XYZ PCA:  MAE = 15.1
            # # ========================================
            # xyz_pcaf -= np.mean(xyz_pcaf)
            # intense = np.mean(np.abs(xyz_pcaf))
            # print("{} {} {} {}:{:.6f}".format(param[0],param[1],param[2], param[3],intense))
            # intensity_load.append(intense)

            # xyz_pcae -= np.mean(xyz_pcae)
            # intense = np.mean(np.abs(xyz_pcae))
            # intensity_no_load.append(intense)

        # plt.scatter(weights, np.asarray(intensity_load)-np.asarray(intensity_no_load))
        # plt.show()

        # print('count:', len(intensity_load))



        intensity_net = np.asarray(intensity_no_load) - np.asarray(intensity_load)
        data = pd.DataFrame({'weights': weights, 'intensity_net': intensity_net})
        data = data.sort_values('intensity_net')
        X = data['intensity_net'].values
        y = data['weights'].values
        kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
        kf.get_n_splits(X, y)

        # print(kf)
        final_GT_list = []
        final_pred_list = []

        for train_index, test_index in kf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index].reshape(-1, 1), X[test_index].reshape(-1, 1)
            y_train, y_test = y[train_index], y[test_index]
            reg = LinearRegression().fit(X_train, y_train)
            # print(reg.score(X_train, y_train))
            # print(reg.coef_)
            pred = reg.predict(X_test)
            # print('pred:', pred)
            # print('GT:', y_test)
            # print('')
            final_GT_list.append(y_test)
            final_pred_list.append(pred)


        pred = np.hstack(final_pred_list)
        gt = np.hstack(final_GT_list)
        # print(pred)
        # print(gt)
        # print('average weights (GT):', np.mean(gt))
        # print('objects:', path)
        print('total number:', len(gt))
        print('min/max weight (GT):', np.min(gt), np.max(gt))
        rmse_val = rmse(pred, gt)
        print("MAE (mean absolute error) is: ", mean_absolute_error(pred, gt))
        print("rms error is: " + str(rmse_val) + '\n\n')


        weights_all += weights
        intensity_load_all += intensity_load
        intensity_no_load_all += intensity_no_load

    print(len(weights_all))
    print(len(intensity_load_all))
    print(len(intensity_no_load_all))

    intensity_net_all = np.asarray(intensity_no_load_all) - np.asarray(intensity_load_all)
    data_all = pd.DataFrame({'weights': weights_all, 'intensity_net': intensity_net_all})
    data_all = data_all.sort_values('intensity_net')
    X = data_all['intensity_net'].values
    y = data_all['weights'].values
    kf = KFold(n_splits=len(weights_all), random_state=1, shuffle=True)
    kf.get_n_splits(X, y)
    final_GT_list = []
    final_pred_list = []

    for train_index, test_index in kf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index].reshape(-1, 1), X[test_index].reshape(-1, 1)
        y_train, y_test = y[train_index], y[test_index]
        reg = LinearRegression().fit(X_train, y_train)
        # print(reg.score(X_train, y_train))
        # print(reg.coef_)
        pred = reg.predict(X_test)
        # print('pred:', pred)
        # print('GT:', y_test)
        # print('')
        final_GT_list.append(y_test)
        final_pred_list.append(pred)


    pred = np.hstack(final_pred_list)
    gt = np.hstack(final_GT_list)
    # print(pred)
    # print(gt)
    # print('average weights (GT):', np.mean(gt))
    print('objects:',paths)
    print('total number:', len(gt))
    print('min/max weight (GT):', np.min(gt), np.max(gt))
    rmse_val = rmse(pred, gt)
    print("MAE (mean absolute error) is: ", mean_absolute_error(pred, gt))
    print("rms error is: " + str(rmse_val))
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylabel("Relative Intensity(m/s^2)", fontsize=28)
    plt.xlabel("Weight(grams)", fontsize=28)
    plt.margins(0,0, tight=True)
    plt.xlim(0,400)
    plt.ylim(-1,6)
    plt.scatter(weights_all, intensity_net_all,linewidths=7)
    plt.savefig("/Users/msi-pc/Downloads/Figure 6.png")
    plt.show()



        
        # print(load_cubic_interp())
        # plt_raw_spec(*nothing())
        # plt_raw_spec(*lemon())
        # plt_raw_spec(*kiwi())
        # plt_raw_spec(*orange1())
        # plt_raw_spec(*orange2())
        # Trial 2, as shown in slide    
        # compare_pca_raw()
        # compare_pca_spec()
        # compare_pca_fft()



