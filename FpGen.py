#!/bin/bash
#requirements
#pip install git+https://github.com/samoturk/mol2vec
#pip install deepchem[tensorflow]
#conda install -c rdkit rdkitactive_fingerprints

import os
import csv
import gzip
import argparse
import pickle
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import RDKFingerprint
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.MolStandardize import rdMolStandardize
from molvs import Standardizer
import tmap as tm
from map4 import MAP4Calculator

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
            #"mol2vec": dc.feat.Mol2VecFingerprint().featurize,
            "rdkit" : dc.feat.RDKitDescriptors().featurize,
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


        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Standardizer().standardize(mol)

            if fp_function == self.fingerprint_functions["ecfp4"]:
                fingerprint = fp_function(mol, 2, nBits=2048)     

            elif fp_function == self.fingerprint_functions["ecfp6"]:
                fingerprint = fp_function(mol, 3, nBits=2048)
            
           # elif fp_function in [self.fingerprint_functions["mol2vec"], self.fingerprint_functions["rdkit"]]:
            #    fingerprint = DataStructs.ConvertToExplicit(fp_function(mol))

            elif fp_function == self.fingerprint_functions["map4"]:
                fingerprint = fp_function(mol)
 
            else:
               fingerprint = fp_function(mol)
            return fingerprint

            
        return None
    
    def map4_fingerprint_to_vector(self, fingerprint):
        """Converts a MAP4 fingerprint to a binary vector."""
        vector = np.zeros(16384, dtype=int)
        for i, bit in enumerate(fingerprint):
            if bit == '1':
                vector[i] = 1
        return vector
    
    def write_fingerprints(self, fingerprints, output_file):
        # Convert tmap.VectorUint objects to lists
        fingerprints = [list(f) if isinstance(f, tm.VectorUint) else f for f in fingerprints]

        with open(output_file, 'wb') as pickle_file:
            pickle.dump(fingerprints, pickle_file)

    
    def process_file(self, input_file, fingerprint_type, file_type):
        fingerprints = []
        # Get the directory name from the input_file path
        directory = os.path.dirname(input_file)
        print(f"Processing {file_type} in directory: {directory}")

        open_func = gzip.open if input_file.endswith(".gz") else open

        with open_func(input_file, 'rb') as input_file:

            if file_type in ["ism", "smi", "smiles"]:
                for line in input_file:
                    smiles = line.strip()
                    #mol = Chem.MolFromSmiles(smiles)
                    fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                    if fingerprint:
                        fingerprints.append(fingerprint)

            elif file_type in ["sdf", "sdf.gz"]:
                suppl = Chem.ForwardSDMolSupplier(input_file)
                for mol in suppl:
                    if mol:
                        smiles = Chem.MolToSmiles(mol)
                        fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                        if fingerprint:
                            fingerprints.append(fingerprint)
                            
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
                                fingerprints.append(fingerprint)

                        processing_molecule = True
                        current_mol_block = b""
                    if processing_molecule:
                        current_mol_block += line.encode('utf-8')
                
                # Process the last molecule in the file
                mol = Chem.MolFromMol2Block(current_mol_block)
                smiles = Chem.MolToSmiles(mol)
                fingerprint = self.generate_fingerprint(smiles, fingerprint_type)
                if fingerprint:
                    fingerprints.append(fingerprint)
            return fingerprints
    def main(self):
        parser = argparse.ArgumentParser(description="Generate fingerprints from SDF.gz files.")
        parser.add_argument("-i", help="Path to the folder containing SDF.gz files.")
        parser.add_argument("-fp", nargs="*", default=["ecfp4"], type=str, choices=self.fingerprint_functions.keys(), help="Type of fingerprint to generate.")
        parser.add_argument("-t", default="ism", type=str, choices=["sdf.gz", "sdf", "ism", "smi", "smiles", "mol2", "mol2.gz"], help="Type of file to process.")
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
                fingerprints = self.process_file(input_folder, fptype, file_type)
                output_file = f"{input_folder}{file}_fingerprints_{fptype}.pkl"
                self.write_fingerprints(fingerprints, output_file)
                print(f"Fingerprints written to: {output_file}")

            elif os.path.isdir(input_folder):
                for root, _, files in os.walk(input_folder):
                    for file in files:
                        if file.endswith(file_type):
                            fingerprints = self.process_file(os.path.join(root, file), fptype, file_type)
                            base = os.path.basename(file)
                            output_file = os.path.join(root, f"{base}_fingerprints_{fptype}.pkl")
                            self.write_fingerprints(fingerprints, output_file)
                            print(f"Fingerprints written to: {output_file}")

if __name__ == "__main__":
    fp_gen = FpGen()
    fp_gen.main()

#example call
#python3 FpGen.py -i test_files/actives_final.ism -fp map4 -t ism

