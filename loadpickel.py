import pickle

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Example usage:
file_path = '/work/ghartimagar/python_project_structure/ionchannel/similarity_matrixMACCSKeys.npy'
loaded_data = load_pkl(file_path)
#print shape of similarity matrix
print(loaded_data.shape)