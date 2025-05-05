import os
import mmh3
import random
import pickle
import argparse
import math
import time
import numpy as np
import pandas as pd
from FpGen import FpGen
from pathlib import Path
from sys import displayhook
from rdkit import Chem
import seaborn as sns
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from sklearn.utils import shuffle
from rdkit.ML.Cluster import Butina
from matplotlib import pyplot as plt
from scipy.spatial.distance import jaccard
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import shapiro, norm, skew, kurtosis, boxcox
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve, average_precision_score, precision_score, recall_score
import multiprocessing
from multiprocessing import Pool, cpu_count
from chemfp import io

class FpCluster:
    def __init__(self):
        self.fp_gen = FpGen()
    # load_fingerprints function takes input folder, fingerprint type and file type as arguments
    # and returns active and decoy fingerprints as lists

    def generate_fingerprints(self, input_folder, fingerprint_type, file_type):
        print("\ngenerate_fingerprints function called with --fpgen argument ...")
        fp_gen = FpGen()

        if not fingerprint_type:
            fingerprint_type = ["ecfp4"]
        if not file_type:
            file_type = ["ism"]

        fingerprints = []
        active_fingerprints = []
        decoy_fingerprints = []

        for fptype in fingerprint_type:
            basename = os.path.basename(input_folder)
            if os.path.isfile(input_folder) and basename.startswith("active") and basename.endswith(file_type):
                print(f"input is an{file_type} file")
                activefp=fp_gen.process_file(input_folder, fptype, file_type)
                active_fingerprints.extend(activefp)
                #save the file as a pickle file
                with open(os.path.join(input_folder, f"actives_{fptype}.pkl"), 'wb') as active_file:
                    pickle.dump(active_fingerprints, active_file)

            elif os.path.isfile(input_folder) and basename.startswith("decoy") and basename.endswith(file_type):
                decoyfp = fp_gen.process_file(input_folder, fptype, file_type)
                decoy_fingerprints.extend(decoyfp)
                #save the file as a pickle file
                with open(os.path.join(input_folder, f"decoys_{fptype}.pkl"), 'wb') as decoy_file:
                    pickle.dump(decoy_fingerprints, decoy_file)

            elif os.path.isfile(input_folder) and "active" not in basename and "decoy" not in basename and basename.endswith(file_type):
                print(f"input is an {file_type} file")
                fp = fp_gen.process_file(input_folder, fptype, file_type)
                fingerprints.extend(fp)
                #save the file as a pickle file
                with open(os.path.join(input_folder, f"{fptype}.pkl"), 'wb') as fingerprint_file:
                    pickle.dump(fingerprints, fingerprint_file)

            elif os.path.isdir(input_folder):
                print(f"input is a directory containing {file_type} files")
                for root, _, files in os.walk(input_folder):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if file.startswith("active") and file.endswith(file_type):
                            activefp = fp_gen.process_file(filepath, fptype, file_type)
                            active_fingerprints.extend(activefp)
                            #save the file as a pickle file
                            with open(os.path.join(root, f"actives_{fptype}.pkl"), 'wb') as active_file:
                                pickle.dump(active_fingerprints, active_file)
                        elif file.startswith("decoy") and file.endswith(file_type):
                            decoyfp = fp_gen.process_file(filepath, fptype, file_type)
                            decoy_fingerprints.extend(decoyfp)
                            #save the file as a pickle file
                            with open(os.path.join(root, f"decoys_{fptype}.pkl"), 'wb') as decoy_file:
                                pickle.dump(decoy_fingerprints, decoy_file)
                        elif "active" not in file and "decoy" not in file and file.endswith(file_type):
                            fp = fp_gen.process_file(filepath, fptype, file_type)
                            fingerprints.extend(fp)
                            #save the file as a pickle file
                            with open(os.path.join(root, f"{fptype}.pkl"), 'wb') as fingerprint_file:
                                pickle.dump(fingerprints, fingerprint_file)

        if active_fingerprints and decoy_fingerprints:
            print ("active and decoy fingerprints generated")
            print(f"Number of active fingerprints: {len(active_fingerprints)}")
            print(f"Number of decoy fingerprints: {len(decoy_fingerprints)}")
            return (active_fingerprints, decoy_fingerprints)
        else:
            print ("other fingerprints generated")
            print(f"Number of fingerprints: {len(fingerprints)}")
            return fingerprints

    
    def load_fingerprints(self, input_folder, fingerprint_types):
        print("\nload_fingerprints function called...")
        active_fingerprints = []  # Initialize lists here
        decoy_fingerprints = []
        fingerprints = []
        for fingerprint_type in fingerprint_types:
            basename = os.path.basename(input_folder)
            print("input is a fingerprint file")
            if os.path.isfile(input_folder) and basename.startswith("active") and basename.endswith(f"{fingerprint_type}.pkl"):
                print("first loop")
                with open(input_folder, 'rb') as active_file:
                    active_fingerprints.extend(pickle.load(active_file))
            elif os.path.isfile(input_folder) and basename.startswith("decoy") and basename.endswith(f"{fingerprint_type}.pkl"):
                print("second loop")
                with open(input_folder, 'rb') as decoy_file:
                    decoy_fingerprints.extend(pickle.load(decoy_file))
            
            elif os.path.isfile(input_folder) and "active" not in basename and "decoy" not in basename and basename.endswith(f"{fingerprint_type}.pkl"):
                print("fourth loop")
                print("input is a fingerprint file")
                with open(input_folder, 'rb') as fingerprint_file:
                    fingerprints.extend(pickle.load(fingerprint_file))
            elif os.path.isdir(input_folder):
                print("third loop")
                print("input is a directory containing fingerprint files")
                for root, _, files in os.walk(input_folder):
                    for file in files:
                        if file.startswith("actives") and file.endswith(f"{fingerprint_type}.pkl"):
                            print("actives")
                            with open(os.path.join(root, file), 'rb') as active_file:
                                active_fingerprints.extend(pickle.load(active_file))
                        elif file.startswith("decoys") and file.endswith(f"{fingerprint_type}.pkl"):
                            print("decoys")
                            with open(os.path.join(root, file), 'rb') as decoy_file:
                                decoy_fingerprints.extend(pickle.load(decoy_file))
                        elif "active" not in file and "decoy" not in file and file.endswith(f"{fingerprint_type}.pkl"):
                            with open(os.path.join(root, file), 'rb') as fingerprint_file:
                                fingerprints.extend(pickle.load(fingerprint_file))

        if active_fingerprints and decoy_fingerprints:
            print(f"Number of active fingerprints: {len(active_fingerprints)}")
            print(f"Number of decoy fingerprints: {len(decoy_fingerprints)}")
            return (active_fingerprints, decoy_fingerprints)
        else:
            print(f"Number of fingerprints: {len(fingerprints)}")
            return fingerprints
    

    def calculate_imbalance(self, active_fingerprints, decoy_fingerprints):
        print ("Calculating imbalance...")
        active_count = len(active_fingerprints)
        decoy_count = len(decoy_fingerprints)
        total_count = active_count + decoy_count

        if active_count > decoy_count:
            imbalance_ratio = active_count / decoy_count
            print(f"Imbalance ratio of active to decoys: {imbalance_ratio:.2f} ({active_count} active compounds, {decoy_count} decoy compounds)")

        elif decoy_count > active_count:
            imbalance_ratio = decoy_count / active_count
            print(f"Imbalance ratio of decoys to active: {imbalance_ratio:.2f} ({decoy_count} decoy compounds, {active_count} active compounds)")

        else:
            print("No imbalance detected.")


    def Similarity_search(self, fp_list, sim_metric:str = "default"):
        """Calculate distance matrix for fingerprint list using Jaccard distance"""
        if sim_metric == "default":
            #metric = self.jaccard_binary #use the jaccard distance function from scipy.spatial.distance #can also use jaccard_binary
            metric = jaccard

        elif sim_metric == "tanimoto":
            dist_mat = self.tanimoto_distance_matrix(fp_list)
            # No need to continue further, so return the dissimilarity matrix
            return dist_mat

        elif sim_metric == "jaccard_like":
            metric = self.jaccard_like

        dist_mat = []
        for i in range(1, len(fp_list)):
            # The jaccard function from scipy.spatial.distance calculates the Jaccard distance, not the Jaccard similarity.
            dist = [metric(fp_list[i], fp_list[j]) for j in range(i)]  #calculates the jaccard dist of ith fingerprint with all the previous fingerprints and subtracts from 1 to get the distance
            dist_mat.extend(dist)
        return dist_mat


    def jaccard_binary(self, x,y):
        """A function for finding the distance between two binary vectors"""
        dist_mat = []
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        similarity = intersection.sum() / float(union.sum())
        dist_mat.extend([1-similarity])
        return dist_mat             


    def tanimoto_distance_matrix(self, fp_list):
        """Calculate distance matrix for fingerprint list"""
        dist_mat = []
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            # Compare the current fingerprint against all the previous ones in the list
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dist_mat.extend([1-x for x in similarities])
        return dist_mat
    

    def jaccard_like(self, a,b):
        """
        Used instead of normal tanimoto for MAP4 and MHFP
        """
        # return 1-float(np.count_nonzero(a == b)) / float(len(a))
        dist_mat = []
        intersection = len(list(set(a).intersection(b)))
        union = (len(a) + len(b)) - intersection
        return 1-(float(intersection) / union)
    
        # #using minhash
        # # Convert integers to bytes-like objects
        # a_bytes = [bytes(str(x), 'utf-8') for x in a]
        # b_bytes = [bytes(str(x), 'utf-8') for x in b]

        # hash_function = mmh3.hash
        # hashed_a = set([hash_function(x) for x in a_bytes])
        # hashed_b = set([hash_function(x) for x in b_bytes])

        # # Calculate Jaccard similarity
        # intersection = len(hashed_a.intersection(hashed_b))
        # union = len(hashed_a.union(hashed_b))
        # if union == 0:
        #     return 0
        # jaccard_similarity = intersection / union
        # return 1 - jaccard_similarity

    def evaluate_distance_dist(self, fingerprints, fingerprint_type, batch_size=10000, path=None):
        """Evaluate the distribution of distances between fingerprints"""
        pairwise_distances = []
        print("fingeprint type", fingerprint_type)
        print(f"\nevaluate_distance_dist function called with {fingerprint_type}...")

        # # Repeat resampling procedure
        # for _ in range(len(fingerprints) // batch_size):
        #     # Randomly select a batch of compounds
        #     batch_indices = np.random.choice(len(fingerprints), size=batch_size, replace=False)
        #     batch_fingerprints = [fingerprints[i] for i in batch_indices]

        #     # Calculate pairwise similarities for the batch
        #     batch_similarities = self.Similarity_search(batch_fingerprints)
        #     pairwise_distances.extend(batch_similarities)

        # Randomly select a subset of fingerprints
        if len(fingerprints) > 0 and len(fingerprints) >= batch_size:
            subset_indices = np.random.choice(len(fingerprints), size=batch_size, replace=False)
            subset_fingerprints = [fingerprints[i] for i in subset_indices]
            print("\nsubset fingerprints", len(subset_fingerprints))

        else:
            subset_fingerprints = fingerprints

        # Calculate the distance matrix for the subset
        if fingerprint_type == "map4":
            print("Using jaccard-like distance metric for map4 fingerprints in evaluate_distance_dist functions")
            subset_distances = self.Similarity_search(subset_fingerprints, sim_metric="jaccard_like")
        else:
            print("Using tanimoto distance metric for other fingerprints in evaluate_distance_dist function")
            subset_distances = self.Similarity_search(subset_fingerprints, sim_metric="tanimoto")
    
        print("\nsubset distances\n", len(subset_distances))
        pairwise_distances.extend(subset_distances)
        negative_values_exist = np.any(np.array(pairwise_distances) < 0)
        if negative_values_exist:
            print("Negative values exist in pairwise distances.")
        else:
            print("No negative values in pairwise distances.")
        output = np.zeros((25))
        # calculate mean, sd, median and percentiles 
        output[0] = np.mean(pairwise_distances)
        output[1] = np.std(pairwise_distances)
        output[2] = np.median(pairwise_distances)
        percentiles = np.arange(5, 100, 5)
        percentiles_values = np.percentile(pairwise_distances, percentiles)
        _, shapiro_pvalue = shapiro(pairwise_distances)
        is_normal = shapiro_pvalue > 0.05
        is_normal = int(is_normal)
        output[3] = is_normal
        output[4] = skew(pairwise_distances)
        output[5 ] = kurtosis(pairwise_distances)
        output[6:25] = percentiles_values

        mean_plus_1sd = output[0] + output[1]
        mean_plus_2sd = output[0] + 2 * output[1]
        mean_plus_3sd = output[0] + 3 * output[1]

        print ("---------------------percentages....................")
        # Check the specified percentages
        print("pairwise distances within mean plus 1 sd", np.sum(pairwise_distances >= mean_plus_1sd) / len(pairwise_distances) * 100)
        print("pairwise distances within mean plus 2 sd",np.sum(pairwise_distances >= mean_plus_2sd) / len(pairwise_distances) * 100)
        print("pairwise distances within mean plus 3 sd", np.sum(pairwise_distances >= mean_plus_3sd) / len(pairwise_distances) * 100)

        # #self.plot_distribution(pairwise_distances, save_path=path)
        # plt.figure(figsize=(12, 6))
        # # plt.subplot(1, 2, 1)
        # plt.hist(pairwise_distances, bins=50, color='blue', alpha=0.7)
        # plt.title('Original Distribution')
        # # plt.show()
        # print("output", len(output))
        print ("output", output)

        return output, pairwise_distances
    
    def save_sim_df(self, output_box: np.ndarray, fp_names: str, path: str, verbose: bool = True) -> None:
        """Saves similarity stats array as csv

        Args:
            output_box:     matrix (S,20) of all similarity vectors
            fp_names:       fingerprint names
            path:           path where to save the .csv file
            verbose:        whether to print the saving directory

        Returns:
            None
        """
        #create names for all collected statistics
        index_names = []
        index_names.append("Mean")
        index_names.append("STD")
        index_names.append("Median")
        index_names.append("Is normal")
        index_names.append("Skewness before")
        index_names.append("Kurtosis before")
        percentiles = np.arange(5, 100, 5)
        index_names.extend([f"Percentile: {percentile}" for percentile in percentiles])
        #print("output box 0", output_box[0])
        #print("output box 1", output_box[1])
        # create and save dataframe
        df = pd.DataFrame(data=output_box, index=index_names, columns=[fp_names])
        df.to_csv(path)
        if verbose:
            print(f"[sim_search]: Saving similarity stats as {path}")

    #clustering molecules
    def cluster_fingerprints(self, input_foder, fingerprints, fingerprint_type="ecfp4", cutoff=0.4):
        """Cluster fingerprints
        Parameters:
            fingerprints
            cutoff: threshold for the clustering
            fingerprint_type: type of fingerprint used
        """
        print(f"\ncluster_fingerprints function called for {fingerprint_type}...")
        # Record the start time for Similarity_search
        start_time_similarity_search = time.time()

        if fingerprint_type == "map4":
            print("Using jaccard-like distance metric for clustering map4 fingerprints")
            distance_matrix = self.Similarity_search(fingerprints, sim_metric="jaccard_like")
            print("shape of distance matrix", len(distance_matrix))

        else:
            print("Using tanimoto distance metric for clustering other fingerprints")
            distance_matrix = self.Similarity_search(fingerprints, sim_metric="tanimoto")
            print("len of distance matrix", len(distance_matrix))

            dis = np.array(distance_matrix)
            print("shape of distance matrix", dis.shape)    

        #save the distance matrix into a file for each fingerprint type into the input folder
        # np.save(os.path.join(input_foder, f"{fingerprint_type}_distance_matrix.npy"), distance_matrix)
        # Record the end time for Similarity_search
        end_time_similarity_search = time.time()

        # Calculate the elapsed time for Similarity_search
        elapsed_time_similarity_search = end_time_similarity_search - start_time_similarity_search
        print(f"Elapsed time for Similarity_search: {elapsed_time_similarity_search:.4f} seconds")

        # Now cluster the data with the implemented Butina algorithm:

        # Record the start time for Butina.ClusterData
        start_time_butina_cluster = time.time()

        clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
        clusters = sorted(clusters, key=len, reverse=True)

        # Record the end time for Butina.ClusterData
        end_time_butina_cluster = time.time()

        # Calculate the elapsed time for Butina.ClusterData
        elapsed_time_butina_cluster = end_time_butina_cluster - start_time_butina_cluster
        print(f"Elapsed time for Butina.ClusterData: {elapsed_time_butina_cluster:.4f} seconds")
        return clusters


    def save_cluster_plot(self, clusters, cutoff, output_folder):
        # Plot the size of the clusters
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.set_title(f"Threshold: {cutoff:3.1f}")
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Number of molecules")
        ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw = 5)
        # Save the plot
        output_path = os.path.join(output_folder, f"cluster_sizes_{cutoff:.1f}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
        # Print the path to the saved image
        print(f"Cluster sizes plot for cutoff {cutoff:.1f} saved at: {output_path}")
        plt.close(fig)
        return output_path
    
    # Sort the molecules within a cluster based on their similarity
    # to the cluster center and sort the clusters based on their size
    def sort_and_select_clusters(self, clusters, fingerprints, fingerprint_type="ecfp4"):
        """Sort clusters by size and select representative compounds"""
        sorted_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue  # Singletons
            # else:
            # extract the fingerprints of the molecules in the cluster center
            cluster_fingerprints = [fingerprints[i] for i in cluster]
            # Sort the fingerprints by their bit counts in decending order

                   # Sort the fingerprints by their bit counts
            if fingerprint_type == "map4":
                #print("no need for getNumOnBits for map4")
                sorted_fingerprints = cluster_fingerprints #cannot sort them according to the number of bits in the fingerprint
            
            else:
                #print("getting getNumOnBits for other fingerprints")
                sorted_fingerprints = sorted(cluster_fingerprints, key=lambda fp: fp.GetNumOnBits(), reverse=True)

            #sorted_fingerprints = sorted(cluster_fingerprints, key=lambda fp: fp.GetNumOnBits(), reverse=True)

            # Similarity of all cluster members to the cluster center
            if fingerprint_type == "map4":
                similarities = [1-self.jaccard_like(sorted_fingerprints[0], fp) for fp in sorted_fingerprints[1:]]
            else:
                similarities = DataStructs.BulkTanimotoSimilarity(sorted_fingerprints[0], sorted_fingerprints[1:])
            # Add index of the molecule to its similarity (centroid excluded!)
            similarities = list(zip(similarities, cluster[1:]))
            # Sort in descending order by similarity
            similarities.sort(reverse=True)
            # Save cluster size and index of molecules in clusters_sort
            sorted_clusters.append((len(similarities), [i for _, i in similarities]))
            # Sort in descending order by cluster size
        sorted_clusters.sort(reverse=True)
        return sorted_clusters


    def select_final_molecules(self, cluster_centers, sorted_clusters, fingerprints, max_total=10000):
        """Select final diverse set of molecules from clusters."""
        selected_molecules = cluster_centers.copy()
        index = 0

        print("len fingerprints", len(fingerprints))
        if len(fingerprints) <=20000:
            max_total = len(fingerprints)
        elif len(fingerprints) > 20000 and len(fingerprints) <= 50000:
            max_total = math.floor(len(fingerprints)/2)
        elif len(fingerprints) > 50000 and len(fingerprints) < 100000:
            max_total = math.floor(len(fingerprints)/3.5)
        elif len(fingerprints) >= 100000 and len(fingerprints) <= 150000:
            max_total = math.floor(len(fingerprints)/5)

        elif len(fingerprints) > 150000 and len(fingerprints) < 250000:
            max_total = math.floor(len(fingerprints)/8)

        else:
            max_total = math.floor(len(fingerprints)/10)

        # max_total = 1000
        print("max total", max_total)
        pending = max_total - len(selected_molecules)
        print("pending", pending)
        
        while pending > 0 and index < len(sorted_clusters):
            #print("entered while loop")
            # Take indices of sorted clusters
            tmp_cluster = sorted_clusters[index][1]
            # If the first cluster is > 10 big then take exactly 10 compounds
            # if sorted_clusters[index][0] > 50:
            #     num_compounds = int(0.2*len(tmp_cluster)) + 1
            #     # print("cluster size > 10")
            if sorted_clusters[index][0] > 10:
                num_compounds = 10
            else:
                # print("cluster size < 10")
                num_compounds = int(0.5 * len(tmp_cluster)) + 1
            if num_compounds > pending:
                # print("num compounds > pending")
                num_compounds = pending
            selected_molecules += [fingerprints[i] for i in map(int, tmp_cluster[:num_compounds])]
            index += 1
            pending = max_total - len(selected_molecules)

            
        return selected_molecules

    def combine_and_shuffle_data(self, input_foder, fingerprint, active_fingerprints, selected_molecules):
        """Combine, label, and shuffle fingerprints."""
        # Step 1: Create labels
        active_labels = np.ones(len(active_fingerprints), dtype=int)
        decoy_labels = np.zeros(len(selected_molecules), dtype=int)
        # Step 2: Combine fingerprints and labels
        combined_fingerprints = np.concatenate([active_fingerprints, selected_molecules])
        combined_labels = np.concatenate([active_labels, decoy_labels])
        # Step 3: Shuffle the combined data
        combined_data = list(zip(combined_fingerprints, combined_labels))
        np.random.shuffle(combined_data)
        combined_fingerprints, combined_labels = zip(*combined_data)
        # Step 4: Convert to numpy arrays
        combined_fingerprints = np.array(combined_fingerprints)
        combined_labels = np.array(combined_labels)

        # save the results for machine learning

        output_folder = os.path.join(input_foder, "combined_data")
        #create the output folder if it does not exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        np.save(os.path.join(output_folder, f"{fingerprint}_combined_fingerprints.npy"), combined_fingerprints)
        np.save(os.path.join(output_folder, f"{fingerprint}_combined_labels.npy"), combined_labels)

        return combined_fingerprints, combined_labels
    

    def train_and_evaluate_classifier(self, combined_fingerprints, combined_labels, model_type, result_file_path, fingerprint_type, decoy_flag=True, target = None):
        print(f"\ntrain_and_evaluate_classifier function called for {fingerprint_type}...")
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
        best_classifier = RandomForestClassifier(random_state=42, **best_params, n_jobs=10) if model_type == "rf" else XGBClassifier(random_state=42, **best_params, n_jobs=4)

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
        parser = argparse.ArgumentParser(description="Combine active and decoy fingerprints, cluster, and undersample.")
        parser.add_argument("-i", "--input_folder", required=True, help="Path to the folder containing subfolders with different fingerprints.")
        parser.add_argument("-fp", "--fingerprint", nargs='+', required=True, choices=["ecfp4", "ecfp6", "atompair", "torsion", "pubchem", "maccs", "avalon", "pattern", "layered", "mol2vec", "rdkit", "map4"], help="Type of fingerprint(s) to combine and label.")
        parser.add_argument("-m", "--model", default=["rf"], nargs='+', choices=["rf", "xg", "light"], help="Type of  model to train.")
        parser.add_argument("-t", "--file_type", default="ism", choices=["sdf.gz", "sdf", "ism", "smi", "mol2", "mol2.gz"], help="Type of file to process.")
        parser.add_argument("-fpgen", "--fpgen", action="store_true",  help="Use this flag to skip the confirmation prompt and overwrite existing files.")

        args = parser.parse_args()
        input_folder = args.input_folder
        fingerprint_types = args.fingerprint
        model_type = args.model
        file_type = args.file_type
        generatefp = args.fpgen
        for fingerprint in fingerprint_types:
            if generatefp:
                fingerprints = self.generate_fingerprints(input_folder, fingerprint, file_type)
            else:
                fingerprints = self.load_fingerprints(input_folder, fingerprint)

            if len(fingerprints) ==2:
                decoy_flag = True
                active_fingerprints = fingerprints[0]
                decoy_fingerprints = fingerprints[1]
                print("No. of active fingerprints", len(active_fingerprints))
                print("No. of decoy fingerprints", len(decoy_fingerprints))

                #calculate imbalance ratio
                imbalance_ratio = len(active_fingerprints)/len(decoy_fingerprints)
                print("imbalance ratio before clustering and selecting molecules", imbalance_ratio)

                #print("decoy fingerprints", decoy_fingerprints)
                stats_mat1, distance_mat= self.evaluate_distance_dist(decoy_fingerprints, fingerprint, batch_size=10000)
                print("stats mat1", stats_mat1)
                self.save_sim_df(stats_mat1, fingerprint, os.path.join(input_folder, f"{fingerprint}_distance_stats1.csv"))  
                #Cluster fingerprints
                clusters = self.cluster_fingerprints(input_folder, decoy_fingerprints, fingerprint_type=fingerprint, cutoff=0.3)
                # Give a short report about the numbers of clusters and their sizes
                num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
                num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
                num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
                num_clust_g100 = sum(1 for c in clusters if len(c) > 100)
                num_clust_g500 = sum(1 for c in clusters if len(c) > 500)
                num_clust_g1000 = sum(1 for c in clusters if len(c) > 1000)
                num_clust_g5000 = sum(1 for c in clusters if len(c) > 5000)
                ##########################################################
                # i = 0
                # for c in clusters:
                #     i += 1
                # print("Number of clusters:", i)
                ##########################################################
                print("# clusters with only 1 compound: ", num_clust_g1)
                print("# clusters with >5 compounds: ", num_clust_g5)
                print("# clusters with >25 compounds: ", num_clust_g25)
                print("# clusters with >100 compounds: ", num_clust_g100)
                print("# clusters with >500 compounds: ", num_clust_g500)
                print("# clusters with >1000 compounds: ", num_clust_g1000)
                print("# clusters with >5000 compounds: ", num_clust_g5000)
                # Get the cluster center of each cluster (first molecule in each cluster)
                cluster_centers = [decoy_fingerprints[c[0]] for c in clusters] 
                # How many cluster centers/clusters do we have?
                print("Number of cluster centers:", len(cluster_centers))
                #sorting and selecting clusters
                sorted_clusters = self.sort_and_select_clusters(clusters, decoy_fingerprints, fingerprint_type=fingerprint)
                #selecting final molecules
                selected_molecules = self.select_final_molecules(cluster_centers, sorted_clusters, decoy_fingerprints)
                print("No. of selected fingerprints", len(selected_molecules))
                #calculate imbalance ratio
                imbalance_ratio = len(active_fingerprints)/len(selected_molecules)
                print("imbalance ratio after clustering and selecting molecules", imbalance_ratio)
                # Combine, label, and shuffle fingerprints
                # combined_fingerprints, combined_labels = self.combine_and_shuffle_data(input_folder, fingerprint,active_fingerprints, selected_molecules)
                #print("Combined fingerprints and labels shuffled.", np.array(combined_fingerprints))
                #combine and shuffle fingerprints
                combined_fingerprints, combined_labels = self.combine_and_shuffle_data(input_folder, fingerprint, active_fingerprints, selected_molecules)
                print ("shape of combined fingerprints", combined_fingerprints.shape)
                print ("shape of combined labels", combined_labels.shape)
                # #train and evaluate classifier
                # for model in model_type:
                #     result_file_path = os.path.join(input_folder, f"{model}_results.csv")
                #     self.train_and_evaluate_classifier(combined_fingerprints, combined_labels, model, result_file_path, fingerprint_type=fingerprint, decoy_flag=decoy_flag)
            else:
                #set a flag for decoy fingerprints
                decoy_flag = False
                fingerprints = np.array(fingerprints)
                toxlabels = pd.read_csv("/work/ghartimagar/python_project_structure/tox21/tox21_labels.csv")

                for model in model_type:
                        for target in toxlabels.columns:
                            print("Fitting %s" % target)
                            target_values = toxlabels[target]   
                            toxlabels_na_removed = target_values.notna().values
                            #fingerprints after removing NAs
                            fingerprints_na_removed = fingerprints[toxlabels_na_removed]
                            target_values_na_removed = target_values[toxlabels_na_removed]
                            #calling the train and evaluate classifier function
                            result_file_path = os.path.join(input_folder, f"{model}_results.csv")
                            self.train_and_evaluate_classifier(fingerprints_na_removed, target_values_na_removed, model, result_file_path, fingerprint_type=fingerprint, decoy_flag=decoy_flag,target=target)

    #########################################################################################################################################
    # def generate_fp_parallel(self, input_folders, fingerprint_types, file_types):
    #     """create main function for running the script in parallel for generating multiple fingerprints"""
    #     print("\nGenerating fingerprints using multiprocessing...")

    #     with Pool(16) as pool:
    #         results = pool.starmap(self.generate_fingerprints, [(folder, fptype, ftype) for folder in input_folders for fptype in fingerprint_types for ftype in file_types])
    #     return results
    
    # def load_fingerprints_parallel(self, input_folders, fingerprint_types):
    #     print("\nload_fingerprints_parallel function called with multiprocessing...")
    #     with Pool() as pool:
    #         results = pool.starmap(self.load_fingerprints, [(folder, fptype) for folder in input_folders for fptype in fingerprint_types])
    #     return results
    ###########################################################################################################################################
                            
    def multi_main(self, input_folder, fingerprint_types, model_type, file_type="ism", generatefp=False):
        """create main function for running the script in parallel for generating multiple fingerprints"""
        print("\nGenerating fingerprints using multiprocessing...")

        for fingerprint in fingerprint_types:
            if generatefp == True:
                fingerprints = self.generate_fingerprints(input_folder, fingerprint_types, file_type)
            else:
                fingerprints = self.load_fingerprints(input_folder, fingerprint_types)

            if len(fingerprints) ==2:
                decoy_flag = True
                active_fingerprints = fingerprints[0]
                decoy_fingerprints = fingerprints[1]
                print("No. of active fingerprints", len(active_fingerprints))
                print("No. of decoy fingerprints", len(decoy_fingerprints))
                
                #calculate imbalance ratio
                imbalance_ratio = len(active_fingerprints)/len(decoy_fingerprints)
                print("imbalance ratio before clustering and selecting molecules", imbalance_ratio)

                #print("decoy fingerprints", decoy_fingerprints)
                stats_mat1, distance_mat= self.evaluate_distance_dist(decoy_fingerprints, fingerprint, batch_size=10000)
                print("stats mat1", stats_mat1)
                self.save_sim_df(stats_mat1, fingerprint, os.path.join(input_folder, f"{fingerprint}_distance_stats1.csv"))  
                #Cluster fingerprints
                clusters = self.cluster_fingerprints(input_folder, decoy_fingerprints, fingerprint_type=fingerprint, cutoff=0.3)
                # Give a short report about the numbers of clusters and their sizes
                num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
                num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
                num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
                num_clust_g100 = sum(1 for c in clusters if len(c) > 100)
                num_clust_g500 = sum(1 for c in clusters if len(c) > 500)
                num_clust_g1000 = sum(1 for c in clusters if len(c) > 1000)
                num_clust_g5000 = sum(1 for c in clusters if len(c) > 5000)
                ##########################################################
                # i = 0
                # for c in clusters:
                #     i += 1
                # print("Number of clusters:", i)
                ##########################################################
                print("# clusters with only 1 compound: ", num_clust_g1)
                print("# clusters with >5 compounds: ", num_clust_g5)
                print("# clusters with >25 compounds: ", num_clust_g25)
                print("# clusters with >100 compounds: ", num_clust_g100)
                print("# clusters with >500 compounds: ", num_clust_g500)
                print("# clusters with >1000 compounds: ", num_clust_g1000)
                print("# clusters with >5000 compounds: ", num_clust_g5000)
                # Get the cluster center of each cluster (first molecule in each cluster)
                cluster_centers = [decoy_fingerprints[c[0]] for c in clusters] 
                # How many cluster centers/clusters do we have?
                print("Number of cluster centers:", len(cluster_centers))
                #sorting and selecting clusters
                sorted_clusters = self.sort_and_select_clusters(clusters, decoy_fingerprints, fingerprint_type=fingerprint)
                #selecting final molecules
                selected_molecules = self.select_final_molecules(cluster_centers, sorted_clusters, decoy_fingerprints)
                print("No. of selected fingerprints", len(selected_molecules))
                #calculate imbalance ratio
                imbalance_ratio = len(active_fingerprints)/len(selected_molecules)
                print("imbalance ratio after clustering and selecting molecules", imbalance_ratio)
                # Combine, label, and shuffle fingerprints
                # combined_fingerprints, combined_labels = self.combine_and_shuffle_data(input_folder, fingerprint,active_fingerprints, selected_molecules)
                #print("Combined fingerprints and labels shuffled.", np.array(combined_fingerprints))
                #combine and shuffle fingerprints
                combined_fingerprints, combined_labels = self.combine_and_shuffle_data(input_folder, fingerprint, active_fingerprints, selected_molecules)
                print("shape of combined fingerprints", combined_fingerprints.shape)
                print("size of combined labels", len(combined_labels))

                ##############################################################################################################################################################                # #train and evaluate classifier
                # for model in model_type:
                #     result_file_path = os.path.join(input_folder, f"{model}_results.csv")
                #     self.train_and_evaluate_classifier(combined_fingerprints, combined_labels, model, result_file_path, fingerprint_type=fingerprint, decoy_flag=decoy_flag)
                ################################################################################################################################################################

            else:
                #set a flag for decoy fingerprints
                decoy_flag = False
                fingerprints = np.array(fingerprints)
                toxlabels = pd.read_csv("/work/ghartimagar/python_project_structure/tox21/tox21_labels.csv")

                for model in model_type:
                        for target in toxlabels.columns:
                            print("Fitting %s" % target)
                            target_values = toxlabels[target]   
                            toxlabels_na_removed = target_values.notna().values
                            #fingerprints after removing NAs
                            fingerprints_na_removed = fingerprints[toxlabels_na_removed]
                            target_values_na_removed = target_values[toxlabels_na_removed]
                            #calling the train and evaluate classifier function
                            result_file_path = os.path.join(input_folder, f"{model}_results.csv")
                            self.train_and_evaluate_classifier(fingerprints_na_removed, target_values_na_removed, model, result_file_path, fingerprint_type=fingerprint, decoy_flag=decoy_flag,target=target)

if __name__ == "__main__":
    fp_processor = FpCluster()
    #fp_processor.main()

    # we will use multiprocessing to perform similarity search and clustering in parallel for all fingerprints for different datasets   
    fingerprint_types = ["ecfp4",] #"ecfp6", "atompair", "torsion", "pubchem", "maccs", "avalon", "pattern", "layered",  "map4"]
    input_folders= ["/work/ghartimagar/python_project_structure/subset/ionchannel", "/work/ghartimagar/python_project_structure/subset/gpcr", 
                    "/work/ghartimagar/python_project_structure/subset/kinase", "/work/ghartimagar/python_project_structure/subset/nuclear", 
                    "/work/ghartimagar/python_project_structure/subset/protease",]
    # input_folders= ["/work/ghartimagar/python_project_structure/testfiles/gpcr",]
    input_folders= ["/work/ghartimagar/python_project_structure/ionchannel",]

    ##########################################################################################################################################
    tic = time.time()
    process_list = []
    for folder in input_folders:
        for fp in fingerprint_types:
            print("folder", folder)
            print("fingerprint", fp)
            #now call multiprocessing module for parallel processing
            p = multiprocessing.Process(target=fp_processor.multi_main, args=(folder, [fp], ["rf"], "ism", False))
            p.start()
            process_list.append(p)

    for process in process_list:
        process.join()

    toc = time.time()
    print("Elapsed time for training and evaluation: {toc-tic:.4f} seconds")

    ############################################################################################################################################

    # file_types =["ism"]

    ############################################################################################################################################    # generated_res = fp_processor.generate_fp_parallel(input_folders, fingerprint_types, file_types)
    # print("generated res", generated_res)
    # # Load fingerprints in parallel
    # loaded_results = fp_processor.load_fingerprints_parallel(input_folders, fingerprint_types)
    # print("Loaded Results:", loaded_results)

    ############################################################################################################################################

    # input_foler = "/work/ghartimagar/python_project_structure/subset/ionchannel"
    # #input_foder = "/work/ghartimagar/python_project_structure/testfiles/gpcr"
    # fingerprint_type = ["ecfp4"]# Add the desired fingerprint types
    # file_type = "ism"  # Specify the file typecd
    # a, d = fp_processor.load_fingerprints(input_folder, fingerprint_type)
    # #a,d = fp_processor.load_fingerprints(input_folder, fingerprint_type)
    # print("Fingerprints loaded...")
    # print(f"Number of active fingerprints: {len(a)}")
    # print(f"Number of decoy fingerprints: {len(d)}")  
    # print("Calculating imbalance...")
    # print(fp_processor.calculate_imbalance(a, d))  
    # #print(calculate_imbalance(a, d))
    # #print(a)s
    # #print(tanimoto_distance_matrix(a))
    # print("\n")
    # #print(Similarity_search(a, sim_metric="jaccard-like"))
    # # Calculate Tanimoto distance matrix
    
    # for fingerprint in fingerprint_type:
    #     print("fingerprint", fingerprint)
    #     if fingerprint == "map4":
    #         print("Using jaccard-like similarity metric for map4 fingerprints")
    #         distance_matrix = fp_processor.Similarity_search(d, sim_metric="jaccard_like")
    #         #print dist matrix values that are greater than 0.8
    #         distance_matrix = np.array(distance_matrix)
    #         print("distance matrix", len(distance_matrix[distance_matrix > 0.8]))
    #     else:
    #         print("Using tanimoto similarity metric for other fingerprints")
    #         distance_matrix = fp_processor.Similarity_search(d, sim_metric="tanimoto")
    #         distance_matrix = np.array(distance_matrix)
    #         print("distance matrix", len(distance_matrix[distance_matrix > 0.8]))

    #     stats_mat1, distance_mat= fp_processor.evaluate_distance_dist(d, fingerprint_type=fingerprint, batch_size=5000, path=os.path.join(input_folder, f"{fingerprint}_distance_stats1.png"))

    #     print(fp_processor.calculate_distribution(distance_mat))
    #     fp_processor.plot_dist(distance_mat)
    #     fp_processor.calculate_statistics(distance_mat)

    #     distance_mat = np.array(distance_mat)
    #     print("-------------------------------------------------------------")
    #     print("Similarity values above 0.8", len(distance_mat[distance_mat > 0.8]))
    #     print("-------------------------------------------------------------")
    #     fp_processor.save_sim_df(stats_mat1, fingerprint_type, os.path.join(input_folder, f"{fingerprint}_distance_stats1.csv"))
    #     # Run the clustering procedure for the dataset

    #     #CLUSTERING
    #     clusters = fp_processor.cluster_fingerprints(d, fingerprint_type=fingerprint)   
    #     print("--------------------------------------------------------------")
    #     print("no. of clusters",len(clusters))
    #     # Give a short report about the numbers of clusters and their sizes
    #     num_clust_g1 = sum(1 for c in clusters if len(c) == 1)
    #     num_clust_g5 = sum(1 for c in clusters if len(c) > 5)
    #     num_clust_g25 = sum(1 for c in clusters if len(c) > 25)
    #     num_clust_g100 = sum(1 for c in clusters if len(c) > 100)
    #     print("total  clusters: ", len(clusters))
    #     print("clusters with only 1 compound: ", num_clust_g1)
    #     print("clusters with >5 compounds: ", num_clust_g5)
    #     print("clusters with >25 compounds: ", num_clust_g25)
    #     print("clusters with >100 compounds: ", num_clust_g100)

    #     # Use the 'Agg' backend to save plots as image files
    #     plt.switch_backend('Agg')
    #     # Plot the size of the clusters
    #     fig, ax = plt.subplots(figsize=(15, 4))
    #     ax.set_xlabel("Cluster index")
    #     ax.set_ylabel("Number of molecules")
    #     ax.bar(range(1, len(clusters) + 1), [len(c) for c in clusters], lw=5)
    #     # Save the plot as an image file
    #     output_folder = input_folder
    #     output_path = os.path.join(output_folder, "cluster_sizes.png")
    #     plt.savefig(output_path)
    #     # Print the path to the saved image
    #     print(f"Cluster sizes plot saved at: {output_path}\n")
    #     #print(clusters)
    #    # print ( "no. of elements in clusters1", len(clusters[0]))
    #     #print ( "no. of elements in clusters2", len(clusters[1]), "\n")

    #     #CHOOSING THE FINAL LIST OF COMPOUNDS RESULTING IN DIVERSE SUBSET
    #     cluster_centers = [d[c[0]] for c in clusters] 
    #     # How many cluster centers/clusters do we have?
    #     print("Number of cluster centers:", len(cluster_centers))
    #     # Sort the molecules within a cluster based on their similarity
    #     sorted_clust = fp_processor.sort_and_select_clusters(clusters, d, fingerprint_type=fingerprint)
    #     print ("sorted clusters", len(sorted_clust))
    #     selected_molecules = fp_processor.select_final_molecules(cluster_centers, sorted_clust, d, max_total=1000)
    #     print("No. of selected fingerprints", len(selected_molecules))
    #     #print("selected fingerprints", selected_molecules)

    #     #calculate imbalance ratio
    #     imbalance_ratio = len(a)/len(selected_molecules)
    #     print("imbalance ratio after clustering and selecting molecules", imbalance_ratio)

    #     # # Combine, label, and shuffle fingerprints
    #     combined_fingerprints, combined_labels = fp_processor.combine_and_shuffle_data(a, selected_molecules)
    #     print("No. of combined fingerprints", len(combined_fingerprints))
    #     print("combined fingerprints", combined_fingerprints)
        

    #     # # Step 4: Split the data into training and testing sets
    #     train_fingerprints, test_fingerprints, train_labels, test_labels = train_test_split(
    #         combined_fingerprints, combined_labels, test_size=0.2, random_state=42
    #     )
    #     # Step 5: Train a RandomForest classifier
    #     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    #     rf_classifier.fit(train_fingerprints, train_labels)
    #     # Step 6: Evaluate the classifier on the testing set
    #     predictions = rf_classifier.predict(test_fingerprints)
    #     probabilities = rf_classifier.predict_proba(test_fingerprints)[:, 1]  # Probabilities of positive class for AUC
    #     # Calculate metrics
    #     accuracy = accuracy_score(test_labels, predictions)
    #     precision = precision_score(test_labels, predictions)
    #     recall = recall_score(test_labels, predictions)
    #     f1 = f1_score(test_labels, predictions)
    #     roc_auc = roc_auc_score(test_labels, probabilities)
    #     pr_auc = average_precision_score(test_labels, probabilities)
    #     # Print the metrics
    #     print(f"Accuracy: {accuracy:.2f}")
    #     print(f"Precision: {precision:.2f}")
    #     print(f"Recall: {recall:.2f}")
    #     print(f"F1 Score: {f1:.2f}")
    #     print(f"AUC-ROC: {roc_auc:.2f}")
    #     print(f"AUC-PR: {pr_auc:.2f}")
