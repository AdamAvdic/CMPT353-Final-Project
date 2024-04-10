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
    print("Analysis (stairs) Steven")
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
    plt.title('Steven: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')
    
    xy_ground_data['label'] = 'flat'    
    xy_walking_data = pd.concat([xy_stair_data, xy_ground_data])   
    ML_classifier(xy_walking_data[['xfreq', 'yfreq']].values, xy_walking_data['label'].values)
	
    plt.show()

    #HENRY
    print("Analysis (stairs) Henry")
    right_h = analyzePeaks('Data/Henry', 'r', 1, 'euclid')  # Assuming r2 is a specific file
    left_h = analyzePeaks('Data/Henry', 'l', 1, 'euclid')   # Assuming l1 is a specific file
    
    # Concatenate and label data for Henry's flat ground analysis
    flat_ground_data_h = pd.concat([right_h, left_h])
    flat_ground_data_h['label'] = 'flat'
    
    # Analysis for Henry's stairs folder
    right_s_h = analyzePeaks('Data/Henry/stairs', 'r', 2, 'euclid')
    left_s_h = analyzePeaks('Data/Henry/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Henry
    stair_data_h = pd.concat([right_s_h, left_s_h])
    stair_data_h['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Henry, main folder
    xy_right_h = xy_peak_pairs('Data/Henry', 'r', 1)
    xy_left_h = xy_peak_pairs('Data/Henry', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_h = pd.concat([xy_right_h, xy_left_h])
    xy_ground_data_h['label'] = 'flat'
    
    # X vs Y frequency comparisons for Henry, stairs folder
    xy_right_s_h = xy_peak_pairs('Data/Henry/stairs', 'r', 2)
    xy_left_s_h = xy_peak_pairs('Data/Henry/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_h = pd.concat([xy_right_s_h, xy_left_s_h])
    xy_stair_data_h['label'] = 'stairs'
    
    plt.figure()
    plt.plot(xy_ground_data_h.xfreq, xy_ground_data_h.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_h.xfreq, xy_stair_data_h.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Henry: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Henry, labeling for classification
    xy_ground_data_h['label'] = 'flat'
    xy_walking_data_h = pd.concat([xy_stair_data_h, xy_ground_data_h])
    
    # Machine Learning classifier applied to Henry's data
    print("Henry - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_h[['xfreq', 'yfreq']].values, xy_walking_data_h['label'].values)
	
    plt.show()

    #ADAM
    right_a = analyzePeaks('Data/Adam', 'r', 1, 'euclid')  # Assuming r2 is a specific file for Adam
    left_a = analyzePeaks('Data/Adam', 'l', 1, 'euclid')   # Assuming l1 is a specific file for Adam
    
    # Concatenate and label data for Adam's flat ground analysis
    flat_ground_data_a = pd.concat([right_a, left_a])
    flat_ground_data_a['label'] = 'flat'
    
    # Analysis for Adam's stairs folder
    right_s_a = analyzePeaks('Data/Adam/stairs', 'r', 2, 'euclid')
    left_s_a = analyzePeaks('Data/Adam/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Adam
    stair_data_a = pd.concat([right_s_a, left_s_a])
    stair_data_a['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Adam, main folder
    xy_right_a = xy_peak_pairs('Data/Adam', 'r', 1)
    xy_left_a = xy_peak_pairs('Data/Adam', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_a = pd.concat([xy_right_a, xy_left_a])
    xy_ground_data_a['label'] = 'flat'
    
    # X vs Y frequency comparisons for Adam, stairs folder
    xy_right_s_a = xy_peak_pairs('Data/Adam/stairs', 'r', 2)
    xy_left_s_a = xy_peak_pairs('Data/Adam/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_a = pd.concat([xy_right_s_a, xy_left_s_a])
    xy_stair_data_a['label'] = 'stairs'
    
    # Plotting and classification for Adam, flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_a.xfreq, xy_ground_data_a.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_a.xfreq, xy_stair_data_a.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Adam: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Adam, labeling for classification
    xy_ground_data_a['label'] = 'flat'
    xy_walking_data_a = pd.concat([xy_stair_data_a, xy_ground_data_a])
    
    # Machine Learning classifier applied to Adam's data
    print("Adam - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_a[['xfreq', 'yfreq']].values, xy_walking_data_a['label'].values)
	
    plt.show()

    #DANIEL
    right_d = analyzePeaks('Data/Daniel', 'r', 1, 'euclid')  # Assuming r2 is a specific file for Daniel
    left_d = analyzePeaks('Data/Daniel', 'l', 1, 'euclid')   # Assuming l1 is a specific file for Daniel
    
    # Concatenate and label data for Daniel's flat ground analysis
    flat_ground_data_d = pd.concat([right_d, left_d])
    flat_ground_data_d['label'] = 'flat'
    
    # Analysis for Daniel's stairs folder
    right_s_d = analyzePeaks('Data/Daniel/stairs', 'r', 2, 'euclid')
    left_s_d = analyzePeaks('Data/Daniel/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Daniel
    stair_data_d = pd.concat([right_s_d, left_s_d])
    stair_data_d['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Daniel, main folder
    xy_right_d = xy_peak_pairs('Data/Daniel', 'r', 1)
    xy_left_d = xy_peak_pairs('Data/Daniel', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_d = pd.concat([xy_right_d, xy_left_d])
    xy_ground_data_d['label'] = 'flat'
    
    # X vs Y frequency comparisons for Daniel, stairs folder
    xy_right_s_d = xy_peak_pairs('Data/Daniel/stairs', 'r', 2)
    xy_left_s_d = xy_peak_pairs('Data/Daniel/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_d = pd.concat([xy_right_s_d, xy_left_s_d])
    xy_stair_data_d['label'] = 'stairs'
    
    # Plotting and classification for Daniel, flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_d.xfreq, xy_ground_data_d.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_d.xfreq, xy_stair_data_d.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Daniel: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Daniel, labeling for classification
    xy_ground_data_d['label'] = 'flat'
    xy_walking_data_d = pd.concat([xy_stair_data_d, xy_ground_data_d])
    
    # Machine Learning classifier applied to Daniel's data
    print("Daniel - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_d[['xfreq', 'yfreq']].values, xy_walking_data_d['label'].values)
	
    plt.show()
    
    #KELLY
    right_k = analyzePeaks('Data/Kelly', 'r', 1, 'euclid')  # Assuming r2 is a specific file for Kelly
    left_k = analyzePeaks('Data/Kelly', 'l', 1, 'euclid')   # Assuming l1 is a specific file for Kelly
    
    # Concatenate and label data for Kelly's flat ground analysis
    flat_ground_data_k = pd.concat([right_k, left_k])
    flat_ground_data_k['label'] = 'flat'
    
    # Analysis for Kelly's stairs folder
    right_s_k = analyzePeaks('Data/Kelly/stairs', 'r', 2, 'euclid')
    left_s_k = analyzePeaks('Data/Kelly/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Kelly
    stair_data_k = pd.concat([right_s_k, left_s_k])
    stair_data_k['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Kelly, main folder
    xy_right_k = xy_peak_pairs('Data/Kelly', 'r', 1)
    xy_left_k = xy_peak_pairs('Data/Kelly', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_k = pd.concat([xy_right_k, xy_left_k])
    xy_ground_data_k['label'] = 'flat'
    
    # X vs Y frequency comparisons for Kelly, stairs folder
    xy_right_s_k = xy_peak_pairs('Data/Kelly/stairs', 'r', 2)
    xy_left_s_k = xy_peak_pairs('Data/Kelly/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_k = pd.concat([xy_right_s_k, xy_left_s_k])
    xy_stair_data_k['label'] = 'stairs'
    
    # Plotting and classification for Kelly, flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_k.xfreq, xy_ground_data_k.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_k.xfreq, xy_stair_data_k.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Kelly: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Kelly, labeling for classification
    xy_ground_data_k['label'] = 'flat'
    xy_walking_data_k = pd.concat([xy_stair_data_k, xy_ground_data_k])
    
    # Machine Learning classifier applied to Kelly's data
    print("Kelly - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_k[['xfreq', 'yfreq']].values, xy_walking_data_k['label'].values)
	
    plt.show()
    
    #CRYSTAL
    right_c = analyzePeaks('Data/Crystal', 'r', 1, 'euclid')  # Assuming 'r2' is a specific file for Crystal
    left_c = analyzePeaks('Data/Crystal', 'l', 1, 'euclid')   # Assuming 'l1' is a specific file for Crystal
    
    # Concatenate and label data for Crystal's flat ground analysis
    flat_ground_data_c = pd.concat([right_c, left_c])
    flat_ground_data_c['label'] = 'flat'
    
    # Analysis for Crystal's stairs folder
    right_s_c = analyzePeaks('Data/Crystal/stairs', 'r', 2, 'euclid')
    left_s_c = analyzePeaks('Data/Crystal/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Crystal
    stair_data_c = pd.concat([right_s_c, left_s_c])
    stair_data_c['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Crystal, main folder
    xy_right_c = xy_peak_pairs('Data/Crystal', 'r', 1)
    xy_left_c = xy_peak_pairs('Data/Crystal', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_c = pd.concat([xy_right_c, xy_left_c])
    xy_ground_data_c['label'] = 'flat'
    
    # X vs Y frequency comparisons for Crystal, stairs folder
    xy_right_s_c = xy_peak_pairs('Data/Crystal/stairs', 'r', 2)
    xy_left_s_c = xy_peak_pairs('Data/Crystal/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_c = pd.concat([xy_right_s_c, xy_left_s_c])
    xy_stair_data_c['label'] = 'stairs'
    
    # Plotting and classification for Crystal, flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_c.xfreq, xy_ground_data_c.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_c.xfreq, xy_stair_data_c.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Crystal: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Crystal, labeling for classification
    xy_ground_data_c['label'] = 'flat'
    xy_walking_data_c = pd.concat([xy_stair_data_c, xy_ground_data_c])
    
    # Machine Learning classifier applied to Crystal's data
    print("Crystal - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_c[['xfreq', 'yfreq']].values, xy_walking_data_c['label'].values)
	
    plt.show()

    #JEFFERY
    right_j = analyzePeaks('Data/Jeffery', 'r', 1, 'euclid')  # Assuming 'r2' is a specific file for Jeffery
    left_j = analyzePeaks('Data/Jeffery', 'l', 1, 'euclid')   # Assuming 'l1' is a specific file for Jeffery
    
    # Concatenate and label data for Jeffery's flat ground analysis
    flat_ground_data_j = pd.concat([right_j, left_j])
    flat_ground_data_j['label'] = 'flat'
    
    # Analysis for Jeffery's stairs folder
    right_s_j = analyzePeaks('Data/Jeffery/stairs', 'r', 2, 'euclid')
    left_s_j = analyzePeaks('Data/Jeffery/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Jeffery
    stair_data_j = pd.concat([right_s_j, left_s_j])
    stair_data_j['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Jeffery, main folder
    xy_right_j = xy_peak_pairs('Data/Jeffery', 'r', 1)
    xy_left_j = xy_peak_pairs('Data/Jeffery', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_j = pd.concat([xy_right_j, xy_left_j])
    xy_ground_data_j['label'] = 'flat'
    
    # X vs Y frequency comparisons for Jeffery, stairs folder
    xy_right_s_j = xy_peak_pairs('Data/Jeffery/stairs', 'r', 2)
    xy_left_s_j = xy_peak_pairs('Data/Jeffery/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_j = pd.concat([xy_right_s_j, xy_left_s_j])
    xy_stair_data_j['label'] = 'stairs'
    
    # Plotting and classification for Jeffery, flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_j.xfreq, xy_ground_data_j.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_j.xfreq, xy_stair_data_j.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Jeffery: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenate XY data for Jeffery, labeling for classification
    xy_ground_data_j['label'] = 'flat'
    xy_walking_data_j = pd.concat([xy_stair_data_j, xy_ground_data_j])
    
    # Machine Learning classifier applied to Jeffery's data
    print("Jeffery - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_j[['xfreq', 'yfreq']].values, xy_walking_data_j['label'].values)
	
    plt.show()
    
    #ELMA
    right_e = analyzePeaks('Data/Elma', 'r', 1, 'euclid')  # Assuming 'r2' is a specific file indication for Elma
    left_e = analyzePeaks('Data/Elma', 'l', 1, 'euclid')   # Assuming 'l1' is a specific file indication for Elma
    
    # Concatenate and label data for Elma's flat ground analysis
    flat_ground_data_e = pd.concat([right_e, left_e])
    flat_ground_data_e['label'] = 'flat'
    
    # Analysis for Elma's stairs folder
    right_s_e = analyzePeaks('Data/Elma/stairs', 'r', 2, 'euclid')
    left_s_e = analyzePeaks('Data/Elma/stairs', 'l', 2, 'euclid')
    
    # Concatenate and label stair data for Elma
    stair_data_e = pd.concat([right_s_e, left_s_e])
    stair_data_e['label'] = 'stairs'
    
    # X vs Y frequency comparisons for Elma, main folder
    xy_right_e = xy_peak_pairs('Data/Elma', 'r', 1)
    xy_left_e = xy_peak_pairs('Data/Elma', 'l', 1)
    
    # Concatenate XY data for flat ground comparison
    xy_ground_data_e = pd.concat([xy_right_e, xy_left_e])
    xy_ground_data_e['label'] = 'flat'
    
    # X vs Y frequency comparisons for Elma, stairs folder
    xy_right_s_e = xy_peak_pairs('Data/Elma/stairs', 'r', 2)
    xy_left_s_e = xy_peak_pairs('Data/Elma/stairs', 'l', 2)
    
    # Concatenate XY data for stairs comparison
    xy_stair_data_e = pd.concat([xy_right_s_e, xy_left_s_e])
    xy_stair_data_e['label'] = 'stairs'
    
    # Plotting and classification for Elma, comparing flat ground vs stairs based on XY frequency data
    plt.figure()
    plt.plot(xy_ground_data_e.xfreq, xy_ground_data_e.yfreq, 'go', label='ground')
    plt.plot(xy_stair_data_e.xfreq, xy_stair_data_e.yfreq, 'bo', label='stairs')
    plt.legend()
    plt.title('Elma: Stairs vs Flat Ground')
    plt.xlabel('xfreq')
    plt.ylabel('yfreq')

    # Concatenating XY data for Elma, preparing for classification
    xy_ground_data_e['label'] = 'flat'
    xy_walking_data_e = pd.concat([xy_stair_data_e, xy_ground_data_e])
    
    # Applying Machine Learning classifier to Elma's data
    print("Elma - Ground vs Stairs classification based on XY frequency data:")
    ML_classifier(xy_walking_data_e[['xfreq', 'yfreq']].values, xy_walking_data_e['label'].values)
	
    plt.show()



if __name__=='__main__':
    main()