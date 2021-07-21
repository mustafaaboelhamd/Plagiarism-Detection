from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    
    # random state for reproducibility
    random_state=1
    
    # split the training data into training and validation with the percentage of 0.07 of the data
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=.07, random_state=random_state)
    
    # models from which we will choose the best one
    models = [KNeighborsClassifier(3),
              SVC(kernel="linear", C=0.025, random_state=random_state),
              SVC(gamma=2, C=1, random_state=random_state),
              GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
              DecisionTreeClassifier(max_depth=5, random_state=random_state),
              RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=random_state),
              MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
              AdaBoostClassifier(n_estimators=500, random_state=random_state),
              GaussianNB(),
              QuadraticDiscriminantAnalysis()]
    
    # initialize the best score to 0 and the index of the best model to 0
    best_score = 0
    best_model_idx = 0
    
    # model selection process -> based on the best score on validation set
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        score = models[i].score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_model_idx = i

    # assigning best model
    model = models[best_model_idx]    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))