import os
import time
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from FpGen import FpGen
from clust_combine import FpCluster
import lightgbm as lgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, precision_score, recall_score
# Define a function for training and evaluating the model

class Models:
    def __init__(self):
        self.fpgen = FpGen()
        self.fpcluster = FpCluster()
    
    def load_data(self, input_folder, fingerprint_type):
        print(f"\nload_data function called for fingerprint type {fingerprint_type}...")
        start_time = time.time()
        #load the fingerprint file and the label file which are stored in the input folder in .npy format
        fingerprint_file = os.path.join(input_folder, f"{fingerprint_type}_combined_fingerprints.npy")
        label_file = os.path.join(input_folder, f"{fingerprint_type}_combined_labels.npy")
        #load the fingerprint and label files
        combined_fingerprints = np.load(fingerprint_file)
        combined_labels = np.load(label_file)
        #print the shape of the fingerprint and label files
        print(f"Fingerprint shape: {combined_fingerprints.shape}")
        print(f"Label shape: {combined_labels.shape}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for loading data: {elapsed_time:.4f} seconds")
        return combined_fingerprints, combined_labels
        
    def train_and_evaluate_classifier(self, combined_fingerprints, combined_labels, model_type, result_file_path, fingerprint_type, decoy_flag=True, target = None):

        print(f"\ntrain_and_evaluate_classifier function called for fingerprint type {fingerprint_type}...")
        start_time = time.time()
        print ("model type", model_type)
        X_train, X_test, y_train, y_test = train_test_split(combined_fingerprints, combined_labels, test_size=0.2, random_state=42)

        # Define the parameter grid for hyperparameter tuning
        param_grid_rf = {
            'n_estimators': [100, 200, 500, 1000, 1500],
            'max_depth': [None, 3, 6, 10, 15, 20],
            'max_features': ['sqrt', 'log2']}
        # the foollowing is the parameter grid for XGBoost for CV
        param_grid_xgb = {
            'max_depth': [3, 6, 10, 15], #typically between 3 to 15
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4], # between 0.05 to 0.3
            'subsample': np.arange(0.5, 1.0, 0.1), #[0.5 0.6 0.7 0.8 0.9]
            'colsample_bytree': np.arange(0.5, 1.0, 0.1),
            'colsample_bylevel': np.arange(0.5, 1.0, 0.1),
            'n_estimators': [100, 200, 400, 700, 1000, 1500],
            'n_estimators': [100, 200, 400, 700, 1000, 1500],
            #elastic net regularization with xgboost
            'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],  # L1 regularization
            'reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10],  # L2 regularization
            #'gamma': [0, 0.1, 0.2, 0.3, 0.4],  # Minimum loss reduction required to make a further partition on a leaf node
            'seed': [42],}

        #lightgbm parameter grid
        param_grid_lgb = {
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 1.0],
            'max_depth': [1, 5, 10, 15, 20, 25, 20, 30,],
            'num_leaves': [10, 25, 50, 100, 150, 200],
            # eg 0.5 tells ligbm to randomly select 50% of features at the beginning of constructing each tree
            'feature_fraction': [0.1, 0.3, 0.5, 0.7, 0.9], 
            # subsample (or bagging_fraction):percentage of rows used per tree building iteration, 
            # means some rows will be randomly selected for fitting each learner (tree).
            #improves generalization and speed of training
            'subsample': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            #for small dataset 10-50 and for large dataset 100-1000
            'min_data_in_leaf': [10, 20, 30, 50, 100, 200, 500],
            'type': ['gbdt', 'dart', 'goss'],
        }
        # Assuming you're using LightGBM, you can set other parameters as well
        lgb_params = {
        'objective': 'binary',  # or 'binary', 'multiclass', etc., based on your task
        'metric': 'binary_logloss',  # or other appropriate metrics
        'verbose' : -1, } # Suppress warnings      

        if model_type == "rf":
            print("Random Forest called")
            classifier = RandomForestClassifier(random_state=42, n_jobs=4)
            param_grid = param_grid_rf
            # print("Random Forest called")
        elif model_type == "xg":
            classifier = XGBClassifier(random_state=42, n_jobs=4)
            param_grid = param_grid_xgb
            # print("XGBoost called")
        elif model_type == "light":
            classifier = lgb.LGBMClassifier(random_state=42, **lgb_params, n_jobs=4)
            param_grid = param_grid_lgb
            # print("LightGBM called")
        else:
            raise ValueError("Invalid model type. Use 'light', 'rf' or 'xg'.")
        # Perform grid search with 5-fold cross-validation
        grid_search = RandomizedSearchCV(classifier, param_grid, cv=5, n_jobs=6)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Initialize the best classifier with the best parameters
        best_classifier = RandomForestClassifier(random_state=42, **best_params, n_jobs=4) if model_type == "rf" else XGBClassifier(random_state=42, **best_params, n_jobs=4)

        # Fit the best classifier on the training data
        best_classifier.fit(X_train, y_train)

        # Predict on the test data
        y_pred = best_classifier.predict(X_test)
        y_proba = best_classifier.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        average_precision = average_precision_score(y_test, y_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        # Print evaluation metrics
        print("Accuracy:", accuracy)
        print("AUC-ROC:", roc_auc)
        print("F1-Score:", f1)
        print("AUC-PR:", pr_auc)
        print("Average Precision:", average_precision)
        print("recall", recall)
        print("precision", precision)
        # # Plot the ROC curve
        # fpr, tpr, _ = roc_curve(y_test, y_proba)
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        #plt.show()
        # Save results to a text file
        result_dict = {
            "target": target,
            "Fingerprint": fingerprint_type,
            "Accuracy": accuracy,
            "AUC-ROC": roc_auc,
            "F1-Score": f1,
            "AUC-PR": pr_auc,
            "Average Precision": average_precision,
            "Recall": recall,
            "Precision": precision,}
        
        if decoy_flag:
            # Extract the fingerprint from the result_dict
            fp = result_dict["Fingerprint"]
            # Check if the result_file_path already exists
            if os.path.exists(result_file_path):
                # Read the existing CSV file into a DataFrame
                existing_df = pd.read_csv(result_file_path)
                # Check if the fingerprint already exists in the DataFrame
                if fp in existing_df["Fingerprint"].values:
                    # Create a DataFrame from the result_dict
                    new_row_df = pd.DataFrame(result_dict, index=[0])
                    # Find the index of the row with the matching fingerprint
                    index_to_update = existing_df[existing_df["Fingerprint"] == fp].index[0]
                    # Update the existing DataFrame with the new values
                    existing_df.loc[index_to_update] = new_row_df.iloc[0]
                else:
                    # Append the new row to the existing DataFrame
                    existing_df = pd.concat([existing_df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)
                # Write the updated DataFrame back to the CSV file
                existing_df.to_csv(result_file_path, index=False)
            else:
                # If the file doesn't exist, create a new DataFrame and write it to the CSV file
                result_df = pd.DataFrame(result_dict, index=[0])
                result_df.to_csv(result_file_path, index=False)

            #give a print statement to show that the results have been saved
            print(f"Results saved to {result_file_path}")
        
        else:
            # Load existing results if the file exists
            if os.path.exists(result_file_path):
                existing_df = pd.read_csv(result_file_path)
                # Check if there is a matching row with the same target and fingerprint name
                matching_row_index = (existing_df["Fingerprint"] == fingerprint_type) & (existing_df["target"] == target)
                if matching_row_index.any():
                    # Replace the matching row with the new values
                    existing_df.loc[matching_row_index, existing_df.columns] = [result_dict[key] for key in existing_df.columns]
                else:
                    # Append the new row to the existing DataFrame
                    existing_df = pd.concat([existing_df, pd.DataFrame(result_dict, index=[0])], ignore_index=True)
                # Write the updated DataFrame back to the CSV file
                existing_df.to_csv(result_file_path, index=False)
                # Give a print statement to show that the results have been saved
                print(f"Results appended to {result_file_path}")
            else:
                # If the file doesn't exist, create a new DataFrame and write it to the CSV file
                result_df = pd.DataFrame(result_dict, index=[0])
                result_df.to_csv(result_file_path, index=False)
                # Give a print statement to show that the results have been saved
                print(f"Results saved to {result_file_path}")
        
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time for training and evaluation: {elapsed_time:.4f} seconds")

    def main(self):
        # Define the argument parser
        parser = argparse.ArgumentParser(description="Train and evaluate a classifier for a given fingerprint type")
        parser.add_argument("-i", "--input_folder", type=str, required=True, help="Type of fingerprint to use for training and evaluation")
        parser.add_argument("-fp","--fingerprint_types", nargs="+", default="ecfp4", type=str, help="Type of fingerprint to use for training and evaluation")
        parser.add_argument("-m", "--model", default="rf", type=str, help="Type of model to use for training and evaluation")
        parser.add_argument("-o", "--output", default=os.getcwd(),type=str, help="Path to the CSV file to save the results")
        args = parser.parse_args()

        fingerprint_types = args.fingerprint_types
        model_type = args.model
        output = args.output
        input_folder = args.input_folder

        for fingerprint in fingerprint_types:
            #call load_data function
            combined_fingerprints, combined_labels = self.load_data(input_folder, fingerprint)
            #call train_and_evaluate_classifier function
            self.train_and_evaluate_classifier(combined_fingerprints, combined_labels, model_type, os.path.join(output, f"{model_type}_results.csv"), fingerprint, decoy_flag=True, target = None)

    def multi_main(self, input_folder, fingerprint_types, model_types, output, decoy_flag=True, target = None):
        for fingerprint in fingerprint_types:
            #call load_data function
            combined_fingerprints, combined_labels = self.load_data(input_folder, fingerprint)
            for model_type in model_types:
                #call train_and_evaluate_classifier function
                self.train_and_evaluate_classifier(combined_fingerprints, combined_labels, model_type, os.path.join(output, f"{model_type}_results.csv"), fingerprint, decoy_flag=True, target = None)

if __name__ == "__main__":
    # Models().main()
    Models = Models()
   # we will use multiprocessing to perform similarity search and clustering in parallel for all fingerprints for different datasets   
    fingerprint_types = ["ecfp4", "ecfp6", "atompair","torsion", "pubchem", "maccs", "avalon", "pattern", "layered",  "map4"]
    # fingerprint_types = ["torsion", "pubchem", "maccs", "avalon", "pattern", "layered",  "map4"]
    models = ["rf", "xg", "light"]
    print(models[0])
    # input_folders= ["/work/ghartimagar/python_project_structure/testfiles/gpcr/combined_data", "/work/ghartimagar/python_project_structure/gpcr/combined_data" ]

    input_folders= ["/work/ghartimagar/python_project_structure/subset2/ionchannel/combined_data",]
    # input_folders= ["/work/ghartimagar/python_project_structure/subset2/gpcr/combined_data",
    #                 "/work/ghartimagar/python_project_structure/subset2/kinase/combined_data", "/work/ghartimagar/python_project_structure/subset2/nuclear/combined_data", 
    #                 "/work/ghartimagar/python_project_structure/subset2/protease/combined_data","/work/ghartimagar/python_project_structure/subset2/ionchannel/combined_data"]
 
    tic = time.time()
    process_list = []
        
    for fp in fingerprint_types:
        for folder in input_folders:
            print("folder", folder)
            print("fingerprint", fp)
            #now call multiprocessing module for parallel processing
            p = multiprocessing.Process(target=Models.multi_main, args=(folder, [fp], ["rf", "xg", "light"], folder, True, None))
            p.start()
            process_list.append(p)

    for process in process_list:
        process.join()

    toc = time.time()
    print(f"Elapsed time for training and evaluating random forest for {fp} in {os.path.basename(folder)}: {toc-tic:.4f} seconds")



#python3 models.py -i /work/ghartimagar/python_project_structure/testfiles/combined_data/ -fp ecfp4 -m rf -o /work/ghartimagar/python_project_structure/testfiles/combined_data/