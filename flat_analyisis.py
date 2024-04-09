import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, interpolate
from math import sqrt, log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn
seaborn.set()

names_list = ["olivia_left", "olivia_right", "nicole_left", "nicole_right", "tiffany_left", "tiffany_right", "clair_left", "clair_right", "paul_left", "paul_right", "kacey_left", \
"kacey_right", "bryan_left", "bryan_right", "ryan_left", "ryan_right", "timothy_left", "timothy_right", "alex_left", "alex_right", \
"sarah_left", "sarah_right", "trevor_left", "trevor_right", "sage_left", "sage_right", "thomas_left", "thomas_right", "nathan_left", "nathan_right", \
"danial_left", "danial_right", "rebecca_left", "rebecca_right", "anthony_left", "anthony_right"]

names_list_FT = ["olivia_left_FT", "olivia_right_FT", "nicole_left_FT", "nicole_right_FT", "tiffany_left_FT", "tiffany_right_FT", "clair_left_FT", "clair_right_FT", "paul_left_FT", "paul_right_FT", "kacey_left_FT", \
"kacey_right_FT", "bryan_left_FT", "bryan_right_FT", "ryan_left_FT", "ryan_right_FT", "timothy_left_FT", "timothy_right_FT", "alex_left_FT", "alex_right_FT", \
"sarah_left_FT", "sarah_right_FT", "trevor_left_FT", "trevor_right_FT", "sage_left_FT", "sage_right_FT", "thomas_left_FT", "thomas_right_FT", "nathan_left_FT", "nathan_right_FT", \
"danial_left_FT", "danial_right_FT", "rebecca_left_FT", "rebecca_right_FT", "anthony_left_FT", "anthony_right_FT"]

CLASSIFIER_OUTPUT = (
    "Bayesian classifier output: {bayes:.4g}\n"
    "kNN classifier output:      {knn:.4g}\n"
    "SVM classifier output:      {svm:.4g}\n"
)

ML_REGRESSION_OUTPUT = (
    "Linear regression output: {lin_reg:.4g}\n"
)

REGRESSION_OUTPUT = (
    "p-value output:       {pval:.4g}\n"
    "r-value output:       {rval:.4g}\n"
    "r-squared output:     {rsquared:.4g}\n"
    "slope output:         {slope:.4g}\n"
    "intercept output:     {intercept:.4g}\n"
    "OLS summary output:   {summary:}\n"
    "Polynomial coefficients output: {pol_reg}\n"
)

def classfierML_function(X, y):
    X_trainer, X_tester, y_trainer, y_tester = train_test_split(X, y)
    knn = KNeighborsClassifier(n_neighbors=3)
    svc = SVC(kernel="linear")
    bayesian = GaussianNB()

    ML_model_list = [bayesian, knn, svc]

    for i, j in enumerate(ML_model_list):  
        j.fit(X_trainer, y_trainer)

    print(CLASSIFIER_OUTPUT.format(
        bayes = bayesian.score(X_tester, y_tester),
        knn = knn.score(X_tester, y_tester),
        svm = svc.score(X_tester, y_tester),
    ))

def regression_plot_maker(X, y, names, model):
    plt.figure()
    plt.plot(X, y, "b.")
    plt.plot(X, model.predict(X), "g-")
    plt.xlabel("Frequency of steps")
    plt.ylabel(names)
    plt.legend(["Original data", "Linear Regression Line"])
    plt.title("Linear regression ML result for " + names + " Versus Frequency of steps\n Testing Data")
    plt.savefig("ML_regression" + names + ".png")
    plt.close()

def regessionML_function(X, y, names):
    X_trainer, X_tester, y_trainer, y_tester = train_test_split(X, y)

    regression_model = LinearRegression(fit_intercept=True)
    regression_model.fit(X_trainer, y_trainer)
    
    regression_plot_maker(X_tester, y_tester, names, regression_model)
    regression_plot_maker(X_trainer, y_trainer, names, regression_model)
    
    print(ML_REGRESSION_OUTPUT.format(
        lin_reg=regression_model.score(X_tester, y_tester),
    ))

def regressionSTAT_function(X, y, names):

    X_trainer, X_tester, y_trainer, y_tester = train_test_split(X, y)

    regression_model = stats.linregress(X_trainer, y_trainer)

    plt.figure()
    plt.plot(X_tester, y_tester, "b.")
    plt.plot(X_tester, X_tester*regression_model.slope + regression_model.intercept, "r-", linewidth=3)
    plt.xlabel("Frequency of steps")
    plt.ylabel(names)
    plt.legend(["Original Data", "Regression Line"])
    plt.title("Linear regression result for " + names + " Versus Frequency of steps\n Testing Data")
    plt.savefig("lin_regression" + names + ".png")
    plt.close()

    plt.figure()
    plt.plot(X_trainer, y_trainer, "b.")
    plt.plot(X_trainer, X_trainer*regression_model.slope + regression_model.intercept, "r-", linewidth=3)
    plt.xlabel("Frequency of steps")
    plt.ylabel(names)
    plt.legend(["Original Data", "Regression Line"])
    plt.title("Linear regression result for " + names + " Versus Frequency of steps\n Training Data")
    plt.savefig("lin_regression" + names + "_train.png")
    plt.close()

    newX_tester = np.linspace(X_tester.min(), X_tester.max(), len(X_tester))
    newX_trainer = np.linspace(X_trainer.min(), X_trainer.max(), len(X_trainer))

    coefficients = np.polyfit(X, y, 5)
    y_fitter = np.polyval(coefficients, newX_tester)
    y_fitter_trainer = np.polyval(coefficients, newX_trainer)

    plt.figure()
    plt.plot(X_tester, y_tester, "b.")
    plt.plot(newX_tester, y_fitter, "go-")
    plt.xlabel("Frequency of steps")
    plt.ylabel(names)
    plt.legend(["Original Data", "Regression Curve"])
    plt.title("Polynomial regression result for " + names + " Versus Frequency of steps\n Testing Data")
    plt.savefig("poly_regression" + names + ".png")
    plt.close()

    plt.figure()
    plt.plot(X_trainer, y_trainer, "b.")
    plt.plot(newX_trainer, y_fitter_trainer, "go-")
    plt.xlabel("Frequency of steps")
    plt.ylabel(names)
    plt.legend(["Original Data", "Regression Curve"])
    plt.title("Polynomial regression result for " + names + " Frequency of stepsy\n Training Data")
    plt.savefig("poly_regression" + names + "_train.png")
    plt.close()

    data_frame = pd.DataFrame({"y": y, "X": X, "intercept": 1})
    regression_results = sm.OLS(data_frame["y"], data_frame[["X", "intercept"]]).fit()

    print(REGRESSION_OUTPUT.format(
        pval = regression_model.pvalue,
        rval = regression_model.rvalue,
        rsquared = regression_model.rvalue**2,
        slope = regression_model.slope,
        intercept = regression_model.intercept,
        summary = regression_results.summary(),
        pol_reg = coefficients,
    ))

def filter_data_frame(data_frame):
    b_val, a_val = signal.butter(3, 0.1, btype = "lowpass", analog = False)
    return signal.filtfilt(b_val, a_val, data_frame)

def plot_specifications(data_frame, X_axis, out_names, names):
    plt.figure()
    plt.plot(data_frame[X_axis], data_frame[names])
    plt.title("Total Linear values" + names)
    plt.xlabel(X_axis)
    plt.savefig(out_names + "_" + names + ".png")
    plt.close()

def euclidian_w(data_frame):
    distance = sqrt(data_frame["wx"]**2 + data_frame["wy"]**2 + data_frame["wz"]**2)
    return distance

def euclidian_a(data_frame):
    distance = sqrt(data_frame["ax"]**2 + data_frame["ay"]**2 + data_frame["az"]**2)
    return distance

def FT_calulator(data_frame, temp_val, iter, numbers):
    data_frameFT = data_frame.apply(np.fft.fft, axis=0)
    data_frameFT = data_frameFT.apply(np.fft.fftshift, axis=0)
    data_frameFT = data_frameFT.abs()

    FS_val = round(len(temp_val)/(temp_val["time"].iloc[-1]-temp_val["time"].iloc[0])) 
    data_frameFT["freq"] = np.linspace(-FS_val/2, FS_val/2, num=len(temp_val))

    plot_specifications(data_frameFT, "freq", names_list[iter] + "_" + str(numbers), "acceleration")
    plot_specifications(data_frameFT, "freq", names_list[iter] + "_" + str(numbers), "velocity")

    temp_FT_val = data_frameFT[data_frameFT.freq > 0.1]
    acc_ind = temp_FT_val["acceleration"].nlargest(n=1)
    max_indVAL = acc_ind.idxmax()
    avg_freqVAL = data_frameFT.at[max_indVAL, "freq"]

    maxVAL = data_frameFT["acceleration"].nlargest(n=1)
    maxVAL_ind = maxVAL.idxmax()
    data_frameFT.at[maxVAL_ind, "acceleration"] = temp_FT_val["acceleration"].max()

    plot_specifications(data_frameFT, "freq", names_list[iter] + "_transformed_" + str(numbers), "acceleration")
    return data_frameFT, avg_freqVAL


def csv_updater(dataSUM, names):
    dataSUM["freq_1"] = ""
    dataSUM["freq_2"] = ""
    dataSUM = dataSUM.set_index("Participant")
    sensor_dataVALS = {}

    for i in range(len(names)):
        name_string =  "Data/flat_data/" + names[i] + ".csv"
        tempVAL = pd.read_csv(name_string)

        walk_dataVALS = pd.DataFrame(columns=["acceleration", "velocity"])

     
        walk_dataVALS["acceleration"] = tempVAL.apply(euclidian_a, axis=1)
        walk_dataVALS["velocity"] = tempVAL.apply(euclidian_w, axis=1)
        
        filter = walk_dataVALS.apply(filter_data_frame, axis=0)

        half_val = round(filter.shape[0]/2)
        filter1 = filter.iloc[:half_val, :]
        filter2 = filter.iloc[half_val:, :]
        tempVAL1 = tempVAL.iloc[:half_val, :]
        tempVAL2 = tempVAL.iloc[half_val:, :]

        dataFT, avg_freq = FT_calulator(filter, tempVAL, i, 0)
        dataFT1, avg_freq_1 = FT_calulator(filter1, tempVAL1, i, 1)
        dataFT2, avg_freq_2 = FT_calulator(filter2, tempVAL2, i, 2)

        dataSUM.at[names[i], "freq_1"] = avg_freq_1
        dataSUM.at[names[i], "freq_2"] = avg_freq_2

        filter_string = names[i] + "_filt"
        string_FT = names[i] + "_FT"

        sensor_dataVALS[string_FT] = dataFT

    return dataSUM, sensor_dataVALS

def visual_generator(temp_data_frame):
    plt.figure()
    tempFrequncies = temp_data_frame["freq"]
    tempFrequncies.plot.hist(alpha=0.5)
    plt.title("Frequency distribution")
    plt.xlabel("Frequencies (Steps per Second)")
    plt.ylabel("Count")
    plt.savefig("freq_distribution.png")
    plt.close()

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    tempFrequncies = temp_data_frame["weight (kg)"]
    tempFrequncies.plot.hist(alpha=0.5)
    plt.title("Weight distribution")
    plt.xlabel("weight (kg)")
    plt.ylabel("Count")

    f.add_subplot(1, 3, 2)
    tempFrequncies = temp_data_frame["Age"]
    tempFrequncies.plot.hist(alpha=0.5)
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")

    f.add_subplot(1, 3, 3)
    tempFrequncies = temp_data_frame["height (cm)"]
    tempFrequncies.plot.hist(alpha=0.5)
    plt.title("Height distribution")
    plt.xlabel("Height (cm)")
    plt.ylabel("Count")
    plt.savefig("height distributions.png")
    plt.close()

    plt.figure()
    tempGender = temp_data_frame.groupby("gender").aggregate("count")
    tempGender["Subject"].plot.pie(autopct="%.2f", figsize=(6, 6))
    plt.title("Number of Males versus Females")
    plt.savefig("gender_graph.png")
    plt.close()

    plt.figure
    f.add_subplot(1, 3, 2)
    tempLOA = temp_data_frame.groupby("level of activity").aggregate("count")
    tempLOA["Subject"].plot.pie(autopct="%.2f", figsize=(6, 6))
    plt.title("Level of Activity of Subjects")
    plt.savefig("Level_of_Activity_pie.png")
    plt.close()

    plt.figure()
    f.add_subplot(1, 3, 3)
    tempAOC = temp_data_frame.groupby("activity of choice").aggregate("count")
    tempAOC["Subject"].plot.pie(autopct="%.2f", figsize=(6, 6))
    plt.title("Activity of Choice for Subjects")
    plt.savefig("Activity_of_Choice_pie.png")
    plt.close()

def main():
    dataSUM = pd.read_csv("Data/flat_data/flat_summary.csv")

    dataSUM, sensor_dataVALS = csv_updater(dataSUM, names_list)
    
    isNA = pd.isna(dataSUM)
    dataSUM = dataSUM[isNA["gender"] == False]
    dataSUM = dataSUM[dataSUM["freq_1"] != ""]
    dataSUM = dataSUM[dataSUM["freq_2"] != ""]

    tempVAL1 = dataSUM.rename(columns={"freq_1": "freq"})
    tempVAL2 = dataSUM.rename(columns={"freq_2": "freq"})
    temp_data_frame = pd.concat([tempVAL1, tempVAL2], axis=0)
    X = temp_data_frame[["freq"]].values
    
    visual_generator(temp_data_frame)
    
    print("Activity Level:")
    classfierML_function(X, temp_data_frame['level of activity'].values)

    print("Gender:")
    classfierML_function(X, temp_data_frame["gender"].values)

    print("Activity of Choice:")
    classfierML_function(X, temp_data_frame["activity of choice"].values)

    print("Anova value:")
    anova = stats.f_oneway(sensor_dataVALS["olivia_left_FT"].acceleration, sensor_dataVALS["olivia_right_FT"].acceleration, sensor_dataVALS["nicole_left_FT"].acceleration, sensor_dataVALS["nicole_right_FT"].acceleration,\
    sensor_dataVALS["nicole_left_FT"].acceleration, sensor_dataVALS["tiffany_right_FT"].acceleration, sensor_dataVALS["clair_left_FT"].acceleration, sensor_dataVALS["olivia_right_FT"].acceleration, \
    sensor_dataVALS["tiffany_left_FT"].acceleration, sensor_dataVALS["tiffany_right_FT"].acceleration, sensor_dataVALS["clair_left_FT"].acceleration, sensor_dataVALS["clair_right_FT"].acceleration, \
    sensor_dataVALS["paul_left_FT"].acceleration, sensor_dataVALS["paul_right_FT"].acceleration, sensor_dataVALS["kacey_left_FT"].acceleration, sensor_dataVALS["kacey_right_FT"].acceleration, \
    sensor_dataVALS["bryan_left_FT"].acceleration, sensor_dataVALS["bryan_right_FT"].acceleration, sensor_dataVALS["ryan_left_FT"].acceleration, sensor_dataVALS["ryan_right_FT"].acceleration, \
    sensor_dataVALS["timothy_left_FT"].acceleration, sensor_dataVALS["timothy_right_FT"].acceleration, sensor_dataVALS["alex_left_FT"].acceleration, sensor_dataVALS["alex_right_FT"].acceleration, \
    sensor_dataVALS["sarah_left_FT"].acceleration, sensor_dataVALS["sarah_right_FT"].acceleration, sensor_dataVALS["trevor_left_FT"].acceleration, sensor_dataVALS["trevor_right_FT"].acceleration, \
    sensor_dataVALS["sage_left_FT"].acceleration, sensor_dataVALS["sage_right_FT"].acceleration, sensor_dataVALS["thomas_left_FT"].acceleration, sensor_dataVALS["thomas_right_FT"].acceleration, \
    sensor_dataVALS["nathan_left_FT"].acceleration, sensor_dataVALS["nathan_right_FT"].acceleration, sensor_dataVALS["danial_left_FT"].acceleration, sensor_dataVALS["danial_right_FT"].acceleration, \
    sensor_dataVALS["rebecca_left_FT"].acceleration, sensor_dataVALS["rebecca_right_FT"].acceleration, sensor_dataVALS["anthony_left_FT"].acceleration, sensor_dataVALS["anthony_right_FT"].acceleration)
    print(anova.pvalue)

    print("Stat Regression:")
    temp_data_frame = temp_data_frame.sort_values("freq")
    x = temp_data_frame["freq"].apply(float)

    print("Height")
    regressionSTAT_function(x, temp_data_frame["height (cm)"].apply(float), "height (cm)")

    print("Age")
    regressionSTAT_function(x, temp_data_frame["Age"].apply(float), "Age")

    print("Weight (kg)")
    regressionSTAT_function(x, temp_data_frame["weight (kg)"].apply(float), "weight (kg)")
    
    print("ML Regression:")

    print("Height (cm)")
    regessionML_function(X, temp_data_frame["height (cm)"].values, "height (cm)")

    print("Age")
    regessionML_function(X, temp_data_frame["Age"].values, "Age")

    print("Weight (kg)")
    regessionML_function(X, temp_data_frame["weight (kg)"].values, "weight (kg)")

    dataSUM.to_csv("output.csv")

if __name__=='__main__':
    main()