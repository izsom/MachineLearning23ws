import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import time

# Function for feature selection version 2
def select_features(X, y, method, corr_threshold=0.5):
    
    if method is None:
        return X.columns
        
    # Feature selection using chi-squared
    elif method == 'p-value':
        # Remove constant features
        X = X.loc[:, X.apply(pd.Series.nunique) != 1]
        p_values = []
        for feature in X.columns:
            f, p = chi2(X[[feature]], y)
            p_values.append(p)

        p_values = np.array(p_values).reshape(-1)
        p_values = pd.Series(p_values, index=X.columns).sort_values()
        selected_features = p_values[p_values < 0.05].index
    
    # Feature selection using Pearson correlation
    elif method == 'correlation':
        X = X.loc[:, X.apply(pd.Series.nunique) != 1]
        corr = X.corr()
        col_corr = set()

        corr_matrix = np.abs(corr.values)

        n = corr_matrix.shape[0]

        for i in range(n):
            for j in range(i):
                if corr_matrix[i, j] > corr_threshold:
                    colname = X.columns[i]
                    col_corr.add(colname)

        selected_features = [col for col in X.columns if col not in col_corr]
                
    return selected_features

# Function for scaling the data
def scale(X_train, X_test, method):
    
    if method is None:
        return X_train, X_test

    # Scaling using StandardScaler 
    elif method == 'standard':
        scaler = StandardScaler().fit(X_train)

    # Scaling using MinMaxScaler
    elif method == 'minmax':
        scaler = MinMaxScaler().fit(X_train)

    # Scaling using RobustScaler
    elif method == 'robust':
        scaler = RobustScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
                
    return X_train_scaled, X_test_scaled



def evaluate_model(X, Y, feature_selection = None, scaling = 'robust', classifier = RandomForestClassifier(), splitting = "holdout", folds=5):
    """_summary_

    Args:
        X (pandas Dataframe): _description_
        Y (pandas Dataframe): _description_
        feature_selection (string): A method to do feature selection. Default is None. Other options are 'p-value' and 'correlation'
        scaling (string): The sclaing method. Default is 'robust'. Other options are 'standard', none and 'minmax'
        classifier ( sklearn.ensemble): default is RandomForestClassifier()
        splitting (string): default method is 'holdout', other option is cv
        folds (int, optional): Parameter to define the cross validation folds. Defaults to 5.

    Returns:
        f1 score: Using the average weighted f1 score
    """
    if splitting == "cv":
        kf = KFold(n_splits=folds, shuffle=True, random_state=18)
        metric_results = {}
        #do the KFold cross validation
        for train_index, test_index in kf.split(X,Y):
            # split the data into train and test for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]   
            
            # preprocess the data
            # feature selection
            preprocess_start_time = time.time()
            rel_features = select_features(X_train, y_train, feature_selection)
            n_features_orig = X_train.shape[1]
            n_features = len(rel_features)
            # scaling the data
            X_train_preprocessed, X_test_preprocessed = scale(X_train[rel_features], X_test[rel_features], scaling)
            preprocess_end_time = time.time()
            preprocess_time = preprocess_end_time - preprocess_start_time
                
            # train and evaluate the model
            clf = classifier
            train_start_time = time.time()
            clf.fit(X_train_preprocessed, y_train)
            train_end_time = time.time()
            training_time = train_end_time - train_start_time
            
            pred_start_time = time.time()
            Y_pred_fold = clf.predict(X_test_preprocessed)
            pred_end_time = time.time()
            prediction_time = pred_end_time - pred_start_time
                
            # calculate the accuracy, precision, recall and f1 score per fold
            accuracy = metrics.accuracy_score(y_test, Y_pred_fold)
            precision = metrics.precision_score(y_test, Y_pred_fold, average='weighted')
            recall = metrics.recall_score(y_test, Y_pred_fold, average='weighted')
            f1 = metrics.f1_score(y_test, Y_pred_fold, average='weighted')

            # add the scores to the dict
            metric_results.setdefault('accuracy', []).append(accuracy)
            metric_results.setdefault('precision', []).append(precision)
            metric_results.setdefault('recall', []).append(recall)
            metric_results.setdefault('f1', []).append(f1)
            metric_results.setdefault('training_time', []).append(training_time)
            metric_results.setdefault('prediction_time', []).append(prediction_time)
            metric_results.setdefault('preprocess_time', []).append(preprocess_time)
            
            
        # Calculate the mean of the metrics over all folds
        accuracy_mean = np.mean(metric_results['accuracy'])
        precision_mean = np.mean(metric_results['precision'])
        recall_mean = np.mean(metric_results['recall'])
        f1_mean = np.mean(metric_results['f1'])
        training_time_mean = np.mean(metric_results['training_time'])
        prediction_time_mean = np.mean(metric_results['prediction_time'])
        preprocess_time_mean = np.mean(metric_results['preprocess_time'])
        
        # Calculate the standard deviation of f1
        f1_std = np.std(metric_results['f1'])

        return {'classifier': classifier.__class__.__name__, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'preprocess_time': preprocess_time,'training_time': training_time, 'prediction_time': prediction_time}

    elif splitting == "holdout":
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=18)
        
        # preprocess the data
        # feature selection
        preprocess_start_time = time.time()
        rel_features = select_features(X_train, y_train, feature_selection)
        n_features_orig = X_train.shape[1]
        n_features = len(rel_features)
        # scaling the data
        X_train_preprocessed, X_test_preprocessed = scale(X_train[rel_features], X_test[rel_features], scaling)
        preprocess_end_time = time.time()
        preprocess_time = preprocess_end_time - preprocess_start_time


        

        # train and evaluate the model
        clf = classifier
        train_start_time = time.time()
        clf.fit(X_train_preprocessed, y_train)
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        
        pred_start_time = time.time()
        Y_pred = clf.predict(X_test_preprocessed)
        pred_end_time = time.time()
        prediction_time = pred_end_time - pred_start_time

        # calculate the accuracy, precision, recall and f1 score per fold
        accuracy = metrics.accuracy_score(y_test, Y_pred)
        precision = metrics.precision_score(y_test, Y_pred, average='weighted')
        recall = metrics.recall_score(y_test, Y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, Y_pred, average='weighted')

        #return {'classifier': classifier.__class__.__name__, 'splitting': splitting, 'feature_selection': feature_selection, 'scaling': scaling, 'n_features_orig': n_features_orig, 'n_features': n_features, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'preprocess_time': preprocess_time,'training_time': training_time, 'prediction_time': prediction_time}
        return {'classifier': classifier.__class__.__name__, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'preprocess_time': preprocess_time,'training_time': training_time, 'prediction_time': prediction_time}
    
    

def apply_smote(df, target_column, k_neighbors=4, random_state=321):
    """_summary_

    Args:
        df (_type_): target dataframe
        target_column (_type_): target column
        k_neighbors (int, optional): Defaults to 4.
        random_state (int, optional): Defaults to 321.

    Returns:
        pandas dataframes: X_resampled, y_resampled
    """

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    return X_resampled, y_resampled
    