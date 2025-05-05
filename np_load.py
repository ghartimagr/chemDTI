import rdkit
from rdkit import Chem

mol = Chem.MolFromSmiles('c1ccccc1')

rd = Chem.RDkitFingerprint(mol)

print(rd)