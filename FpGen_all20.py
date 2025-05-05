#!/bin/bash

#Implementing automatic pipelie from ism files to  different types of fingerprints using RDkit
# We are given a folder with 102 subfolders, and each subfolder contains 2 kinds of .ism file for smiles : 
#first one for ligands and second one for decoys. We will use the .ism files and 
#create differrent kind of fingerprints from them. Outputs should be in separate files. 
#We should also standardize the moleculesby sanitizing, adding Hs to molecules etc. so that fingerprints are consistent.
#We will construct a FpGen class to generate different types of fingerprints from SMILES strings.
#We will use argparse for argument parsing to call different kind of fingerprints : 
# rdkit (topological), morgan, daylight, pubchem, atom pair, topological torsion, maccs keys, etc.#

#requirements
#pip install git+https://github.com/samoturk/mol2vec
#pip install deepchem[tensorflow]
#conda install -c rdkit rdkit

import os
import csv
import gzip
import argparse
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from molvs import Standardizer
import tmap as tm
from map4 import MAP4Calculator
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints


# from rdkit.Chem.Pharm2D import Generate
# from deepchem.feat import WeaveFeaturizer
# import deepmol as dm
# from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer 

# Define a dictionary mapping fingerprint names to RDKit functions
# Extended-Connectivity Fingerprint (ECFP) is also known as Circular Fingerprint or Circular Morgan Fingerprint
# ECFP4 fingerprints use a radius of 2 to define the circular neighborhood surrounding each atom, while ECFP6 fingerprints use a radius of 3
# The AtomPair fingerprint is a topological fingerprint based on determining the shortest distance between all pairs of atoms within amolecule
# MACCS is a type of substructure key-based fingerprint which uses 166 predefined keys to encode the presence or absence of particular substructures
# The RDKitFP fingerprint is a hashed fingerprint based on the topological atom environment of each atom in a molecule
# it finds all subgraphs in a molecule containing a number of bonds within a predefined range, hashes the subgraphs, 
# and then uses these hashes to generate a bit vector of fixed length
#The LayeredFP uses the same algorithm as the RDKitFP to identify subgraphs, but different bits are set in the final fingerprint
#  based on different “layers” (different atom and bond type definitions).


class FpGen:
    def __init__(self):
        self.fingerprint_functions = {
            #rdkit fingerprints
            "ecfp4": AllChem.GetMorganFingerprintAsBitVect,
            "ecfp6": AllChem.GetMorganFingerprintAsBitVect,
            "atompair": AllChem.GetHashedAtomPairFingerprintAsBitVect,
            "torsion": AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect,
            "pubchem": AllChem.RDKFingerprint,
            "maccs": MACCSkeys.GenMACCSKeys,
            "avalon": pyAvalonTools.GetAvalonFP,
            "pattern": Chem.PatternFingerprint,
            "layered": Chem.LayeredFingerprint,
            "map4" : MAP4Calculator(dimensions=1024).calculate,
            #deepchem fingerprints
            "mol2vec": dc.feat.Mol2VecFingerprint().featurize,
            "rdkit" : dc.feat.RDKitDescriptors().featurize,
            "conv": dc.feat.ConvMolFeaturizer().featurize, #DAGModel, GraphConvModel
            "gconv": dc.feat.MolGraphConvFeaturizer().featurize, #AttentiveFP, GAT, GCN,MPNN, InfoGraph, Infographstar
            #cdk fingerprints
            "standard" : get_fingerprint,
            "extended" : get_fingerprint, 
            "graph" : get_fingerprint,
            "estate" : get_fingerprint,
            "hybridization" : get_fingerprint,
            "klekota-roth" : get_fingerprint, 
            "shortestpath" : get_fingerprint, 
            "cdk-substructure" : get_fingerprint, 
            "circular": get_fingerprint,
            "cdk-atompairs" : get_fingerprint,
            "lingo" : get_fingerprint,
            #rdktypes
            "rdk-maccs" :get_fingerprint,
            "topological-torsion" : get_fingerprint,
            #babel fingerprints
            #babel fingerprint give errors
            "fp2" : get_fingerprint,
            "fp3" : get_fingerprint,
            "fp4" : get_fingerprint,
            "spectrophores" : get_fingerprint,
            #vectorized fingerprints
            #"mol2vec": get_fingerprint
        }

    def generate_fingerprint(self, smiles, fingerprint_type):
        """
        Generate a fingerprint for a given SMILES string using the specified fingerprint type.

        Args:
            smiles (str): The SMILES string of the molecule.
            fingerprint_type (str): The type of fingerprint to generate.

        Returns:
            str: The binary fingerprint representation as a string.
        """
        
        fp_function = self.fingerprint_functions.get(fingerprint_type)

        if fp_function in [self.fingerprint_functions["standard"], self.fingerprint_functions["extended"], self.fingerprint_functions["graph"],
                            self.fingerprint_functions["estate"], self.fingerprint_functions["hybridization"], self.fingerprint_functions["klekota-roth"], 
                            self.fingerprint_functions["shortestpath"], self.fingerprint_functions["cdk-substructure"], self.fingerprint_functions["cdk-atompairs"], 
                            self.fingerprint_functions["lingo"], self.fingerprint_functions["rdk-maccs"], self.fingerprint_functions["topological-torsion"],
                            self.fingerprint_functions["fp2"],self.fingerprint_functions["fp3"],self.fingerprint_functions["fp4"],self.fingerprint_functions["spectrophores"]]:
            fingerprint = fp_function(smiles, fingerprint_type)
            return fingerprint

        #convert smiles to mol object
        mol = Chem.MolFromSmiles(smiles)
        #standardize the molecule
        if mol is not None:
            #Chem.SanitizeMol(mol,sanitizeOps=(Chem.SANITIZE_ALL^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_PROPERTIES))
            
            mol = Standardizer().standardize(mol)
            #standardize the molecule
            # remove Hs, disconnect metal atoms, normalize the molecule, reionize the molecule
            # clean_mol = rdMolStandardize.Cleanup(mol)

            # # if many fragments, get the "parent" (the actual mol we are interested in)
            # parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

            # # try to neutralize molecule
            # uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
            # uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

            # # note that no attempt is made at reionization at this step
            # te = rdMolStandardize.TautomerEnumerator() # idem
            # taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
            # mol = taut_uncharged_parent_clean_mol         

            #get the fingerprint function from the dictionary as specified by the user in the command line

            #Unlike other fingerprints, Morgan fingerprints require a radius parameter, 
            #therefore we need to check if the fingerprint is morgan 
            if fp_function == self.fingerprint_functions["ecfp4"]:
                fingerprint = fp_function(mol, 2)
                # return fingerprint.ToBitString()
                #returns string of fingerprint bits
                return ",".join(str(bit) for bit in fingerprint)
                #fingerprint_list = [int(bit) for bit in fingerprint]
                #return fingerprint_list

            elif fp_function == self.fingerprint_functions["ecfp6"]:
                fingerprint = fp_function(mol, 3)
                return ",".join(str(bit) for bit in fingerprint)
                #fingerprint_list = [int(bit) for bit in fingerprint]
                #return fingerprint_list
                        
            elif fp_function in [self.fingerprint_functions["mol2vec"], self.fingerprint_functions["rdkit"]]:  
                fingerprint_list= fp_function(mol)
                fingerprint_list = fingerprint_list[0]
                fingerprint_str = ",".join(str(bit).replace('\n', '') for bit in fingerprint_list)
                return fingerprint_str
                #fingerprint_list = [float(bit) for bit in fingerprint_list[0]]
                #return fingerprint_list
            
            elif fp_function == self.fingerprint_functions["map4"]:
                fingerprint = fp_function(mol)
                fingerprint_str = ",".join(str(bit).replace('\n', '') for bit in fingerprint)
                return fingerprint_str
                # fingerprint_list = [float(bit) for bit in fingerprint]
                # return fingerprint_list
            
            #Weave convolution takes too long to run, so we will not use it          
            #elif fp_function in [self.fingerprint_functions["conv"],self.fingerprint_functions["wconv"]]:
            # returns list of arrays
            # elif fp_function == self.fingerprint_functions["conv"]: 
            #     features = fp_function(mol)
            #     #fingerprint_list = []
            #     for i, feature in enumerate(features):
            #         #feature_list = [feature.get_atom_features() for feature in features]
            #         # fingerprint_list.append(feature_list)
            #         fingerprint_list = feature.get_atom_features()
            #     # return fingerprint_list
            #     return ",".join(str(bit) for bit in fingerprint_list)

            #returns a row of features for each molecule, but different number of features for each molecule
            elif fp_function == self.fingerprint_functions["conv"]:
                conv_mol_list = []
                conv_mol = fp_function([mol])[0]
                conv_mol_list.append(conv_mol)
                fingerprint_list = []

                for feature in conv_mol_list:
                    fingerprint = feature.get_atom_features()
                    for atom_feature in fingerprint:#
                        fingerprint_list.append(str(bit).replace('\n', '') for bit in atom_feature)#
                
                fingerprint_array = np.array(fingerprint_list)
                fingerprint_str = ",".join([",".join(map(str, row)) for row in fingerprint_array])
                return fingerprint_str
                #returns fp bits : as integers for conv
                # conv_mol_list = []
                # conv_mol = fp_function([mol])[0]
                # conv_mol_list.append(conv_mol)
                # fingerprint_list = []

                # for feature in conv_mol_list:
                #     fingerprint = feature.get_atom_features()
                #     # Convert feature bits to integers
                #     int_feature = [[int(bit) for bit in atom_feature] for atom_feature in fingerprint]
                #     fingerprint_list.append(int_feature)
                    
                # return fingerprint_list

            elif fp_function == self.fingerprint_functions["gconv"]:
                #returns list of arrays : as string
                #unflatten the list of arrays: returns bits as string
                gconv_mol_list = []
                gconv_mol = fp_function([mol])[0]
                gconv_mol_list.append(gconv_mol)
                fingerprint_list = []
                for feature in gconv_mol_list:
                    fingerprint = feature.node_features
                    for atom_feature in fingerprint:
                        fingerprint_list.append(str(bit).replace('\n', '') for bit in atom_feature)
                fingerprint_array = np.array(fingerprint_list)
                fingerprint_str = ",".join([",".join(map(str, row)) for row in fingerprint_array])
                return fingerprint_str
                ##returns bits as integers
                # gconv_mol_list = []
                # gconv_mol = fp_function([mol])[0]
                # gconv_mol_list.append(gconv_mol)
                # fingerprint_list = []
                # for feature in gconv_mol_list:
                #     fingerprint = feature.node_features
                #     for atom_feature in fingerprint:
                #         # Convert feature bits to integers
                #         int_feature = [int(bit) for bit in atom_feature]
                #         fingerprint_list.append(int_feature)
                # return fingerprint_list

            # returns bits as stringsfor gconv
            #returns list of arrays : as string
            # elif fp_function == self.fingerprint_functions["gconv"]:
            #     features = fp_function(mol)
            #     fingerprint_list = []
            #     # for every feature in features, get node features by using feature.node_features function
            #     # and append to fingerprint_list
            #     for i, feature in enumerate(features):
            #         # feature_list = [feature.node_features for feature in features]
            #         # fingerprint_list.append(feature_list)
            #         fingerprint_list = feature.node_features
            #     # return fingerprint_list
            #     return ",".join(str(bit) for bit in fingerprint_list)

            else:
                fingerprint = fp_function(mol)
                return ",".join(str(bit) for bit in fingerprint)
                # fingerprint_list = [int(bit) for bit in fingerprint]
                # return fingerprint_list

            
        return None
    
    def process_file(self, input_file, fingerprint_type, file_type):

        # Get the directory name from the input_file path
        directory = os.path.dirname(input_file)
        print(f"Processing {file_type} in directory: {directory}")

        #output_file = f"{os.path.splitext(input_file)[0]}_{file_type }_{fingerprint_type}_fingerprints.csv"
        output_file = f"{os.path.splitext(input_file)[0]}_{fingerprint_type}_fingerprints.csv"
        open_func = gzip.open if input_file.endswith(".gz") else open

        with open_func(input_file, 'rb') as input_file, open(output_file, 'w') as output:
            writer = csv.writer(output)

            if file_type in ["ism", "smi"]:
                for line in input_file:
                    smiles = line.strip()
                    #mol = Chem.MolFromSmiles(smiles)
                    fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                    if fingerprint:
                        writer.writerow([fingerprint])
            elif file_type in ["sdf", "sdf.gz"]:
                suppl = Chem.ForwardSDMolSupplier(input_file)
                for mol in suppl:
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                        if fingerprint:
                            writer.writerow([fingerprint])
            elif file_type in ["mol2", "mol2.gz"]:
                current_mol_block = b""
                processing_molecule = False
                for line in input_file:
                    line = line.decode('utf-8')  # Decode binary to string
                    if line.strip().startswith("@<TRIPOS>MOLECULE"):
                        # A new molecule is starting
                        if processing_molecule:
                            # Process the previous molecule
                            mol = Chem.MolFromMol2Block(current_mol_block)
                            smiles = Chem.MolToSmiles(mol)
                            fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                            if fingerprint:
                                writer.writerow([fingerprint])

                        processing_molecule = True
                        current_mol_block = b""
                    if processing_molecule:
                        current_mol_block += line.encode('utf-8')
                
                # Process the last molecule in the file
                mol = Chem.MolFromMol2Block(current_mol_block)
                smiles = Chem.MolToSmiles(mol)
                fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                if fingerprint:
                    writer.writerow([fingerprint])

    def main(self):
        parser = argparse.ArgumentParser(description="Generate fingerprints from SDF.gz files.")
        parser.add_argument("-i", help="Path to the folder containing SDF.gz files.")
        parser.add_argument("-fp", nargs="*", default=["ecfp4"], type=str, choices=self.fingerprint_functions.keys(), help="Type of fingerprint to generate.")
        parser.add_argument("-t", default="ism", type=str, choices=["sdf.gz", "sdf", "ism", "smi", "mol2", "mol2.gz"], help="Type of file to process.")
        args = parser.parse_args()

        input_folder = args.i
        fingerprint_type = args.fp
        file_type = args.t

        if not os.path.exists(input_folder):
            raise ValueError(f"Root directory '{input_folder}' does not exist.")

        if not fingerprint_type:
            fingerprint_type = ["ecfp4"]

        for fptype in fingerprint_type:
            if os.path.isfile(input_folder) and input_folder.endswith(file_type):
                self.process_file(input_folder, fptype, file_type)
            elif os.path.isdir(input_folder):
                for root, _, files in os.walk(input_folder):
                    for file in files:
                        if file.endswith(file_type):
                            self.process_file(os.path.join(root, file), fptype, file_type)

if __name__ == "__main__":
    fp_gen = FpGen()
    fp_gen.main()

#example call
#python3 FpGen.py -i test_files/actives_final.ism -fp map4 -t ism