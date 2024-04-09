import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from math import sqrt
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

OUTPUT_TEMPLATE_CLASSIFIER =  (
    'bayesClassifier:   {bayes:.3g}\n'
    'kNNClassifier:     {knn:.3g}\n'
    'SVMClassifier:     {svm:.3g}\n'
)

OUTPUT_TEMPLATE_REGRESS = (
    'linearRegression: {linReg:.2g}\n'
    'polynomialRegression: {polyReg:.3g}\n'
)


# Used for the ML model for prediciting participants gender, height, weight
def ML_classifier(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    bayesModel = GaussianNB()
    knnModel = KNeighborsClassifier(n_neighbors=3)
    svcModel = SVC(kernel = 'linear')

    models = [bayesModel, knnModel, svcModel]

    for i, m in enumerate(models):
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes = bayesModel.score(X_train, y_train),
        knn = knnModel.score(X_train, y_train),
        svm = svcModel.score(X_train, y_train)
    ))

def filterDF(df):
    b, a = signal.butter(3, 0.1, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, df)
    
# walkData is the acceleration data to be filtered and transformed.  
# data is the original dataframe read from csv
def filter_and_fft(walkData, data):
    #Filter the data
    dataFilt = walkData.apply(filterDF, axis=0)
   
    #Take the Fourier Transform of the data
    dataFilt = dataFilt.apply(np.fft.fft, axis=0)
    dataFilt = dataFilt.apply(np.fft.fftshift, axis=0)
    dataFilt = dataFilt.abs()

    #Determine the sampling frequency

    # Samples per second
    Fs = round(len(data)/data.at[len(data)-1, 'time']) 
   
    dataFilt['freq'] = np.linspace(-Fs/2, Fs/2, num=len(data))
    
    return dataFilt

def plotAcc(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['acceleration'])
    plt.title('Total Linear Acceleration')
    plt.xlabel(x_axis)
    plt.show()

def plotVel(df, x_axis, output_name):
    plt.figure()
    plt.plot(df[x_axis], df['velocity'])
    plt.title('Total Angular Velocity')
    plt.xlabel(x_axis)
    plt.savefig(output_name + '_vel.png')
    plt.close()

def euclDistW(df):
    return sqrt(df['wx']**2 + df['wy']**2 + df['wz']**2)

def eculDistA(df):
    return sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    
    
def read_csv(dir, file_ext, n, i):
     
        if dir == 'Data':
            str_name =  dir + '/' + str(i) + file_ext +  '.csv'
        else:
            str_name =  dir + '/' + file_ext + str(i) + '.csv'
			
        return pd.read_csv(str_name)
	
# dir is the file directory
# file_ext is the prefix of the file,
# n is the number of data sets
# mode: ax (acceleration in x), ay (acceleration in y), euclid (euclidian norm of x, y and z)
# extracts the peak of the major spikes in the data 

def analyzePeaks(dir, file_ext, n, mode):
    # extracts the 2 largest peaks that are characteristic of a signal
    importantBlips = pd.DataFrame()
    
    for i in range(1,n+1):
	
        # left and right 13 doesnt exist, skip for now
        if i == 13:
            continue            
            
        data = read_csv(dir, file_ext, n, i)
        
        walkData = pd.DataFrame(columns=['acceleration'])
       
        if mode == 'euclid':
            #Take the Euclidean Norm
            walkData['acceleration'] = data.apply(eculDistA, axis=1)
        elif mode == 'ax':
            walkData[['acceleration']] = data[['ax']]
        elif mode == 'ay':
            walkData[['acceleration']] = data[['ay']]
        else:
            print("error in mode arg for analyzePeaks")
            break
        
        # split data into 2
        startInd = 0
        length = len(walkData)
        for j in range(1, 3):

            walkDataSeg = walkData.iloc[startInd:startInd+int(length/2), :].reset_index().drop(columns=['index'])
            dataSeg = data.iloc[0:len(walkDataSeg), :].reset_index().drop(columns=['index']) # time needs to start from 0 again
            startInd = startInd + int(length/2)

            dataFT = filter_and_fft(walkDataSeg, dataSeg)
            
            # ignore low freq noise
            dataFT = dataFT[dataFT['freq'] > 0.4]
            
            # Get the local max values, keep only the "significant" blips, lets say those above 40% of max blip
            ind = argrelextrema(dataFT.acceleration.values, np.greater)
            localMax = dataFT.acceleration.values[ind]
            localMax = localMax[localMax > 0.5 * localMax.max()]
            importantBlips = pd.concat([importantBlips, dataFT[dataFT['acceleration'].isin(localMax)]])

    return importantBlips

# dir is the file directory
# file_ext is the prefix of the file,
# n is the number of data sets
# Gets the freq of the peak from both x spikes and y spikes, makes a datapoint with them.
# If theres more than one peak in both x and y, a datapoint is made for each combination
def xy_peak_pairs(dir, file_ext, n):
    # extracts the 2 largest peaks that are characteristic of a signal
    xy_peaks = pd.DataFrame(columns = ['xfreq', 'yfreq'])
    
    for i in range(1,n+1):	
        # left and right 5 doesn't exist, skip it
        if i == 5:
            continue            
            
        data = read_csv(dir, file_ext, n, i)
        
        walkData = pd.DataFrame(columns=['ax', 'ay'])       
      
        walkData[['ax']] = data[['ax']]        
        walkData[['ay']] = data[['ay']]        
        
        # split data into 2
        startInd = 0
        length = len(walkData)
        for j in range(1, 3):

            walkDataSeg = walkData.iloc[startInd:startInd+int(length/2), :].reset_index().drop(columns=['index'])
            dataSeg = data.iloc[0:len(walkDataSeg), :].reset_index().drop(columns=['index']) # time needs to start from 0 again
            startInd = startInd + int(length/2)

            dataFT = filter_and_fft(walkDataSeg, dataSeg)
            
            # ignore low freq noise
            dataFT = dataFT[dataFT['freq'] > 0.4]            
            
            # Get the local max values, keep only the "significant" blips, lets say those above 40% of max blip
            indx = argrelextrema(dataFT.ax.values, np.greater)
            indy = argrelextrema(dataFT.ay.values, np.greater)
            
            xlocal_max = dataFT.ax.values[indx]
            xlocal_max = xlocal_max[xlocal_max > 0.5 * xlocal_max.max()]
            ylocal_max = dataFT.ay.values[indy]
            ylocal_max = ylocal_max[ylocal_max > 0.5 * ylocal_max.max()]
            
            xlocal_max_freq = dataFT[dataFT['ax'].isin(xlocal_max)]['freq'].values
            ylocal_max_freq = dataFT[dataFT['ay'].isin(ylocal_max)]['freq'].values
            pairs = np.transpose([np.tile(xlocal_max_freq, len(ylocal_max_freq)), np.repeat(ylocal_max_freq, len(xlocal_max_freq))])
            # Filter out any empty or all-NA entries from xy_peaks before concatenating
            non_empty_xy_peaks = xy_peaks.dropna(axis=1, how='all')
            xy_peaks = pd.concat([non_empty_xy_peaks, pd.DataFrame(data=pairs, columns=['xfreq', 'yfreq'])])


    return xy_peaks

def main():

    # left leg and right leg on flat ground
    right = analyzePeaks('Data/Steven', 'r', 3, 'euclid')
    left = analyzePeaks('Data/Steven', 'l', 3, 'euclid')
    
    plt.plot(right.freq, right.acceleration, 'go', label='right leg')
    plt.plot(left.freq, left.acceleration, 'bo', label='left leg')
    plt.title('Left and Right leg Characteristic Frequencies')
    plt.legend()
    plt.xlabel('freq')
    
    right['label'] = 'right'
    left['label'] = 'left'
    
    flat_ground_data = pd.concat([right, left])
    print("Left leg vs right leg classification:")
    ML_classifier(flat_ground_data[['freq', 'acceleration']].values, flat_ground_data['label'].values)
    
    # left leg and right leg on stairs
    right_s = analyzePeaks('Data/Steven/stairs', 'r', 2, 'euclid')
    left_s = analyzePeaks('Data/Steven/stairs', 'l', 2, 'euclid')
    stair_data = pd.concat([right_s, left_s])
    stair_data['label'] = 'stairs'
    
    plt.figure()
    plt.plot(flat_ground_data.freq, flat_ground_data.acceleration, 'go', label='ground')
    plt.plot(stair_data.freq, stair_data.acceleration, 'bo', label='stairs')    
    plt.legend()    
    plt.title('Stairs vs Flat Ground')
    plt.xlabel('freq')
    
    flat_ground_data['label'] = 'flat'    
    my_walking_data = pd.concat([stair_data, flat_ground_data])
    print("Ground vs Stairs classification")    
    ML_classifier(my_walking_data[['freq', 'acceleration']].values, my_walking_data['label'].values)
    
    
    # x vs y frequency comparisons
    print("x vs y")
    xy_left = xy_peak_pairs('Data/Steven', 'l', 3)
    xy_right = xy_peak_pairs('Data/Steven', 'r', 3)
    
    plt.figure()
    plt.plot(xy_right.xfreq, xy_right.yfreq, 'go', label='right leg')
    plt.plot(xy_left.xfreq, xy_left.yfreq, 'bo', label='left leg')
    plt.title('Left and Right leg Characteristic Frequencies x vs y')
    plt.legend()
    plt.xlabel('freq')
    plt.ylabel('yfreq')
    
    xy_right['label'] = 'right'
    xy_left['label'] = 'left'
    xy_ground_data = pd.concat([xy_right, xy_left])
    ML_classifier(xy_ground_data[['xfreq', 'yfreq']].values, xy_ground_data['label'].values)
    
    # Stairs
    xy_right_s = xy_peak_pairs('Data/Steven/stairs', 'r', 2)
    xy_left_s = xy_peak_pairs('Data/Steven/stairs', 'l', 2)
    xy_stair_data = pd.concat([xy_right_s, xy_left_s])
    xy_stair_data['label'] = 'stairs'
    
    plt.figure()
    plt.plot(xy_ground_data.xfreq, xy_ground_data.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data.xfreq, xy_stair_data.yfreq, 'bo', label='stairs')    
    plt.legend()    
    plt.title('Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')
    
    xy_ground_data['label'] = 'flat'    
    xy_walking_data = pd.concat([xy_stair_data, xy_ground_data])   
    ML_classifier(xy_walking_data[['xfreq', 'yfreq']].values, xy_walking_data['label'].values)
	
    plt.show()


if __name__=='__main__':
    main()