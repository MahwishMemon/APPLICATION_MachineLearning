import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# app heading
st.write("""
         # Explore Machine Learning Classifiers with different data_sets, 
         **and Analysi which one is best for your data_set !**
         """)

# make data sets name in sidebar
dataset_name= st.sidebar.selectbox("Select Dataset", 
                                   ("Iris", "Breast Cancer", "Wine Dataset", "Digits Dataset")
                                   )

# make classifiers name in sidebar
classifier_name= st.sidebar.selectbox("Select Classifier",
                                      ( "KNN", "SVM", "Random Forest")
                                      )

def get_dataset (dataset_name):
    if dataset_name == "Iris":
        data= datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data= datasets.load_breast_cancer()
    elif dataset_name == "Wine Dataset":
        data= datasets.load_wine()
    elif dataset_name == "Digits Dataset":
        data= datasets.load_digits()
    else:
        data= datasets.load_iris()
    X= data.data
    y= data.target
    return X, y

# call the function to get the data set and split it
X, y= get_dataset(dataset_name)

# print shape of data set in app 
st.write("Shape of dataset", X.shape) # shape of data set
st.write("Number of classes", len(np.unique(y))) # number of classes in data set (target)

#add different parameters for each classifier userinput
def add_parameter_ui(classifier_name):
    params= dict() # make a dictionary to store the parameters
    if classifier_name == "KNN":
        K= st.sidebar.slider("K", 1, 15) # k is the number of neighbors
        params["K"]= K # add K to the dictionary
    elif classifier_name == "SVM":
        C= st.sidebar.slider("C", 0.01, 10.0) # C is the regularization parameter
        params["C"]= C # add C to the dictionary
    else:
        max_depth= st.sidebar.slider("max_depth", 2, 15)
        n_estimators= st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"]= max_depth # max_depth is the maximum depth of the tree
        params["n_estimators"]= n_estimators # n_estimators is the number of trees in the forest
    return params
# call the function to get the parameters
Params= add_parameter_ui(classifier_name)

# make classifiers based on classifier name
def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf= KNeighborsClassifier(n_neighbors= params["K"])
    elif classifier_name == "SVM":
        clf= SVC(C= params["C"])
    else:
        clf= RandomForestClassifier(n_estimators= params["n_estimators"],
                                    max_depth= params["max_depth"], random_state=1234)
    return clf

# call the function to get the classifier
clf = get_classifier(classifier_name, Params)


# split the data set into train and test
X_train , X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1234)

# fit the classifier to the data
clf. fit(X_train, y_train)
y_pred= clf.predict(X_test)

# calculate the accuracy
acc= accuracy_score(y_test, y_pred)
st.write (f"Classifier = {classifier_name}")
st.write (f"Accuracy = {acc}")


### plot the data set ### 
pca= PCA(2) # make a PCA object with 2 components
X_projected= pca.fit_transform(X) # fit the data to the PCA object

# slice the data to get 0 and 1 dimensions
x1= X_projected[:, 0]
x2= X_projected[:, 1]
fig= plt.figure()
plt. scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)

