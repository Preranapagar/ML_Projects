###########################################################################
# Required Python Packages
###########################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
###########################################################################
# File Paths
###########################################################################
INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"
###########################################################################
# Headers
###########################################################################
HEADERS = ['CodeNumber','ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize',
           'BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','CancerType']

###########################################################################
# Function name : read_data
# Description : read the data into pandas dataframe
# Input : path of csv file
# Output : Gives the data
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def read_data(path):
    data = pd.read_csv(path)
    return data
###########################################################################
# Function name : get_headers
# Description : dataset headers
# Input : dataset
# Output : returns the headers
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def get_headers(dataset):
    return dataset.columns.values
###########################################################################
# Function name : add_headers
# Description : add headers to dataset
# Input : dataset
# Output : updates dataset
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset
###########################################################################
# Function name : data_file_to_csv
# Description : converts datafile to csv
# Input : nothing
# output : returns csv datafile
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def data_file_to_csv():
    #Headers
    headers = ['CodeNumber','ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize',
           'BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','CancerType']
    #load the dataset into pandas dataframe
    dataset = read_data(INPUT_PATH)
    #add headers to the loaded datset
    dataset = add_headers(dataset, headers)
    #save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH, index = False)
    print("File saved...")

###########################################################################
# Function name : split_dataset
# Description : split the datset with train_percentage
# Input : Dataset with related information
# Output : Dataset after spliting
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def split_dataset(dataset, train_percentage, feature_headers, target_headers):
    train_X, test_X, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_headers], train_size = train_percentage)
    return train_X, test_X, train_y, test_y

###########################################################################
# Function name : handling_missing_values
# Description : Filter missing values from the dataset
# Input : dataset
# Output : dataset with removed missing values
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def handling_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

###########################################################################
#Function name : Random_forest_classifier
# Description : to train the random forest classifier with features and train data
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def Random_forest_classifer(features,target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

###########################################################################
# Function name : datset_statistics
# Description : basic statistics of dataset
# Input : Dataset
# Output : Description of dataset
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################
def dataset_statistics(dataset):
    print(dataset.describe())

###########################################################################
# Function name : main
# Description : main funcion which executes the application
# Author : Prerana Pagar
# Date : 29/10/2023
###########################################################################

def main():
    #load the csv file
    dataset = pd.read_csv(OUTPUT_PATH)

    #Get the basic stats of loaded dataset
    dataset_statistics(dataset)

    #filter missing values
    dataset = handling_missing_values(dataset, HEADERS[6], '?')

    #split dataset
    X_train, X_test, y_train, y_test = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

    #Train and test dataset size deatils
    print("X train shape", X_train.shape)
    print("X test shape", X_test.shape)
    print("Y train shape", y_train.shape)
    print("Y test shape", y_test.shape)

    #create random forest classifier
    trained_model = Random_forest_classifer(X_train,y_train)
    print("Trained model :", trained_model)
    predictions = trained_model.predict(X_test)

    for i in range(0,205):
        print("Actual outcome :: {} and Predicted outcome ::{}".format(list(y_test)[i],predictions[i]))
    
    print("Train Accuracy :", accuracy_score(y_train, trained_model.predict(X_train)))
    print("Test Accuracy :", accuracy_score(y_test, predictions))
    print("Confusion Matrix :\n", confusion_matrix(y_test, predictions))

###########################################################################
#Application starter
###########################################################################
if __name__=="__main__":
    main()

