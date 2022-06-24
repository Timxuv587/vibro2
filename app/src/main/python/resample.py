import os 
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import create_folder, list_files_in_directory


def resample(data_df, time_col_header, sampling_rate, gap_tolerance=np.inf, fixed_time_column=None):
    """
    Resample a pandas DataFrame fix a new sampling rate using linear interpolation.

    Parameters
    ----------
    data_df : data dataframe, contains unixtime column and data column(s)

    time_col_header : string, time column header

    sampling_rate : int
        number of samples per second

    gap_tolerance: int(ms), by default np.inf
        if the distance between target point and either of the neighbors is further than gap_tolerance in millisecond,
        then interpolation is nan
        if gap_tolerance=0, the gap_tolerance rule will not exist

    fixed_time_column: np.array or list, a given array(list) of sampling positions on which the resampling is based, by default None        

    Examples
    --------
    >>> time_col_header = 'unixtime'
    >>> df = pd.DataFrame(np.arange(20).reshape(5,4),columns=['unixtime', 'a', 'b', 'c'])

    >>> unix = np.array([1500000000000,1500000000048,1500000000075,1500000000100,1500000000150])
    >>> df['unixtime'] = unix
    >>> print(df)
            unixtime   a   b   c
    0  1500000000000   1   2   3
    1  1500000000048   5   6   7
    2  1500000000075   9  10  11
    3  1500000000100  13  14  15
    4  1500000000150  17  18  19
    >>> new_sampling_rate = 20
    >>> new_df = resample(df, time_col_header, new_sampling_rate)
    >>> print(new_df)
            unixtime          a          b          c
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000050   5.296295   6.296295   7.296295
    2  1500000000100  13.000000  14.000000  15.000000
    3  1500000000150  17.000000  18.000000  19.000000

    >>> new_sampling_rate = 33
    >>> new_df = resample(df, time_col_header, new_sampling_rate)
    >>> print(new_df)
            unixtime          a          b          c
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000030   3.525238   4.525238   5.525238
    2  1500000000060   6.867554   7.867554   8.867554
    3  1500000000090  11.545441  12.545441  13.545441
    4  1500000000121  14.696960  15.696960  16.696960

    (_note: the 5th unixtime is 1500000000121 instead of 1500000000120, since 5th sampling is 121.21ms away from 1st sampling.
    
    -----
    """

    original_name_order = list(data_df.columns.values)
    unixtime_arr = data_df[time_col_header].values
    delta_t = 1000.0/sampling_rate
    
    data_df = data_df.drop(time_col_header, axis=1)
    data_arr = data_df.values
    names = list(data_df.columns.values)

    n = len(unixtime_arr)
    new_data_list = []
    
    if n<2:
        return

    if fixed_time_column is None:
        #_looping through columns to apply the resampling method for each column
        for c in range(data_arr.shape[1]):
            signal_arr = data_arr[:,c]

            # always take the first timestamp time[0]
            new_signal_list = [signal_arr[0]]
            new_unixtime_list = [unixtime_arr[0]]

            t = unixtime_arr[0] + delta_t
            t_ind_after = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = t_ind_after
                for i in range(i0,n):
                    if  t <= unixtime_arr[i]:#we found the needed time index
                        t_ind_after = i
                        break

                # interpolate in the right interval, gap_tolenance=0 means inf tol,
                if gap_tolerance == 0 or \
                    (abs(unixtime_arr[t_ind_after-1]-unixtime_arr[t_ind_after])<=gap_tolerance):
                    s = interpolate(unixtime_arr[t_ind_after-1], signal_arr[t_ind_after-1], \
                                    unixtime_arr[t_ind_after], signal_arr[t_ind_after], t)
                else:
                    s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                new_signal_list.append(s)
                new_unixtime_list.append(int(t))
                # take step further on time
                t = t + delta_t
                # check the stop condition
                if t > unixtime_arr[-1]:
                    break

            new_data_list.append(new_signal_list)
            new_data_arr = np.transpose(np.array(new_data_list))

        data_df = pd.DataFrame(data = new_data_arr, columns = names)
        data_df[time_col_header] = np.array(new_unixtime_list)

        # change to the original column order
        data_df = data_df[original_name_order]

    else:  # if fixed_time_column not None:
        # looping through columns to apply the resampling method for each column
        for c in range(data_arr.shape[1]):
            signal_arr = data_arr[:,c]
            new_signal_list = []
            new_unixtime_list = []

            i_fixed_time = 0

            t = fixed_time_column[i_fixed_time]
            t_ind_after = 0
            out_of_range = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = t_ind_after
                for i in range(i0,n):
                    if  t <= unixtime_arr[i]:#we found the needed time index
                        t_ind_after = i
                        out_of_range = 0
                        break

                if out_of_range:
                    s = np.nan
                else:
                    # interpolate in the right interval
                    if t_ind_after == 0: # means unixtime_arr[0] > t, there is no element smaller than t
                        s = np.nan
                    elif gap_tolerance == 0 or \
                        (abs(unixtime_arr[t_ind_after-1] - unixtime_arr[t_ind_after]) <= gap_tolerance):
                        s = interpolate(unixtime_arr[t_ind_after-1], signal_arr[t_ind_after-1], \
                                        unixtime_arr[t_ind_after], signal_arr[t_ind_after], t)
                    else:
                        s = np.nan

                # append the new interpolated sample to the new signal and update the new time vector
                new_signal_list.append(s)
                new_unixtime_list.append(int(t))

                # check the stop condition
                if t > unixtime_arr[-1]:
                    break
                # take step further on time
                i_fixed_time += 1

                if i_fixed_time >= len(fixed_time_column):
                    break
                t = fixed_time_column[i_fixed_time]

            new_data_list.append(new_signal_list)
            new_data_arr = np.transpose(np.array(new_data_list))

        data_df = pd.DataFrame(data = new_data_arr, columns = names)
        data_df[time_col_header] = np.array(new_unixtime_list)

        # change to the original column order
        data_df = data_df[original_name_order]
    return data_df


def interpolate(t1, s1, t2, s2, t):
    """
    _interpolates at parameter 't' between points (t1,s1) and (t2,s2)
    """
    if(t1<=t and t<=t2): #we check if 't' is out of bounds (between t1 and t2)
        m = float(s2 - s1)/(t2 - t1)
        b = s1 - m*t1
        return m*t + b
    else:
        return np.nan


def resampling_pandas(df, sampling_freq=20, higher_freq=100, max_gap_sec=1):
    """ _resample unevenly spaced timeseries data linearly by
    first upsampling to a high frequency (short_rate)
    then downsampling to the desired rate.

    _parameters
    ----------
        df:               data_frame
        sampling_freq:    sampling frequency
        max_gap_sec:      if gap larger than this, interpolation will be avoided

    _return
    ------
        result:           data_frame

    _note: _you will need these 3 lines before resampling_pandas() function
    ---------------------------------------------------------------------
        # df['date'] = pd.to_datetime(df['_time'],unit='ms')
        # df = df.set_index(['date'])
        # df.index = df.index.tz_localize('utc').tz_convert(settings.timezone)

    """
    
    # find where we have gap larger than max_gap_sec
    # print(df.index)
    # diff = np.diff(df.index)

    # print(diff)
    idx = np.where(np.greater(np.diff(df.index), 1000))[0]
    start = df.index[idx].tolist()
    stop = df.index[idx + 1].tolist()
    big_gaps = list(zip(start, stop))

    # upsample to higher frequency
    df = df.resample('{}ms'.format(1000/higher_freq)).mean().interpolate()

    # downsample to desired frequency
    df = df.resample('{}ms'.format(1000/sampling_freq)).ffill()

    # remove data inside the gaps
    for start, stop in big_gaps:
        df[start:stop] = None
    df.dropna(inplace=True)

    return df


def resample_folder(in_path, out_path, time_col_header, sampling_rate, gap_tolerance=np.inf):
    """
    creates resampled csv data in directroy for subject

    Parameters
    ----------
    in_path: str
        path for input
    out_path: str
        path to save rasampled output
    time_col_header: str
        header for the time column
    sampling_rate : int
        number of samples per second
    gap_tolerance: int(ms), by default np.inf
        if the distance between target point and either of the neighbors is further than gap_tolerance in millisecond,
        then interpolation is nan
        if gap_tolerance=0, the gap_tolerance rule will not exist
    """
    create_folder(out_path)
    files = list_files_in_directory(in_path)

    for file in files:
        if not file.startswith('.'):
            data_df = pd.read_csv(os.path.join(in_path, file))

            if len(data_df):
                if 'date' in data_df.columns:
                    data_df = data_df.drop(columns=['date'])
                new_df = resample(data_df, time_col_header, sampling_rate, gap_tolerance,
                                  fixed_time_column=None)
                new_df.to_csv(os.path.join(out_path, file), index=None)




def load_cubic_interp_pandas():
    """
    Note: pandas cubic spline has its problem: dataframe.interpolate(method='spline', order=3)
	                             time_tick  acc_X_value  acc_Y_value  acc_Z_value
	time                                                                     
	1970-01-01 00:01:10.006     70.006      0.00853    -0.061354     9.785901
	1970-01-01 00:01:10.007        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.008        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.009        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.010        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.011        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.012        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.013        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.014        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.015        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.016     70.016     -0.00015     0.025140     9.785751
	1970-01-01 00:01:10.017        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.018        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.019        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.020        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.021        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.022        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.023        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.024        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.025        NaN          NaN          NaN          NaN
	                         time_tick  acc_X_value  acc_Y_value  acc_Z_value
	time                                                                     
	1970-01-01 00:01:10.006     70.006     0.008530    -0.061354     9.785901
	1970-01-01 00:01:10.007     70.007     0.021214     0.004320     9.786631
	1970-01-01 00:01:10.008     70.008     0.021216     0.004319     9.786631
	1970-01-01 00:01:10.009     70.009     0.021218     0.004319     9.786631
	1970-01-01 00:01:10.010     70.010     0.021220     0.004319     9.786631
	1970-01-01 00:01:10.011     70.011     0.021222     0.004319     9.786632
	1970-01-01 00:01:10.012     70.012     0.021224     0.004319     9.786632
	1970-01-01 00:01:10.013     70.013     0.021226     0.004319     9.786632
	1970-01-01 00:01:10.014     70.014     0.021228     0.004319     9.786632
	1970-01-01 00:01:10.015     70.015     0.021230     0.004318     9.786632
	1970-01-01 00:01:10.016     70.016    -0.000150     0.025140     9.785751
	1970-01-01 00:01:10.017     70.017     0.021233     0.004318     9.786633
	1970-01-01 00:01:10.018     70.018     0.021235     0.004318     9.786633
	1970-01-01 00:01:10.019     70.019     0.021237     0.004318     9.786633
	1970-01-01 00:01:10.020     70.020     0.021239     0.004318     9.786633
	1970-01-01 00:01:10.021     70.021     0.021241     0.004318     9.786634
	1970-01-01 00:01:10.022     70.022     0.021243     0.004317     9.786634
	1970-01-01 00:01:10.023     70.023     0.021245     0.004317     9.786634
	1970-01-01 00:01:10.024     70.024     0.021247     0.004317     9.786634
	1970-01-01 00:01:10.025     70.025     0.021249     0.004317     9.786635
    input: 
        -- df
    output:
        -- df_resample

    """
    df = pd.read_csv('../test/lemon41gbound-acc.csv')
    df = df[(df['time_tick']<90) & (df['time_tick']>70)]

    # df = df[['time', col]]
    df['time_tick'] = (df['time_tick'].values*1000).astype(int)/1000
    df['time'] = pd.to_datetime(df['time_tick'], unit='s')
    df = df.set_index('time')
    df_resample = df.resample('0.001S').asfreq()
    print(df_resample.head(20))
    df_resample['time_tick'] = df_resample['time_tick'].interpolate(method='spline', order=3)
    df_resample['acc_X_value'] = df_resample['acc_X_value'].interpolate(method='spline', order=3)
    df_resample['acc_Y_value'] = df_resample['acc_Y_value'].interpolate(method='spline', order=3)
    df_resample['acc_Z_value'] = df_resample['acc_Z_value'].interpolate(method='spline', order=3)
    print(df_resample.head(20))
    # df_resample = df_resample.dropna()
    t = df_resample['time_tick'].values
    x = df_resample['acc_X_value'].values
    y = df_resample['acc_Y_value'].values
    z = df_resample['acc_Z_value'].values    
    plt_raw_xyz(t, x, y, z)

    return df_resample
