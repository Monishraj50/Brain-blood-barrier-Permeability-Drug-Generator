from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import ast

RDLogger.DisableLog("rdApp.*")
data = pd.read_excel('./B3DB-main/raw_data/R1/data_formatted_done.xls')
print(data.head())

print(data[data['BBB+/BBB-']=='BBB+'].reset_index()['smiles'].tolist())

data_smiles=[]
sum_smiles = 0
for i in range(1,51):
    data = pd.read_excel('./B3DB-main/raw_data/R{}/data_formatted_done.xls'.format(i))
    data = data[data['BBB+/BBB-']=='BBB+'].reset_index()['smiles'].tolist()
    data_smiles.append(data)
    sum_smiles+=len(data)
    
data = [item for sublist in data_smiles for item in sublist]
data = pd.DataFrame(data)
data.columns=['smiles']


for i,smiles in enumerate(data['smiles']):
    try:
        Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
    except (AttributeError,TypeError):
        print(i)
        print(smiles)
        data.drop(i,inplace=True)

maxi = 0
for i,smiles in enumerate(data['smiles']):
    molecule = Chem.MolFromSmiles(smiles)
    num = molecule.GetNumHeavyAtoms()
    if num>=maxi:
        maxi= num
        
l = []
for smiles in data['smiles']:
    for atom in Chem.MolFromSmiles(smiles).GetAtoms():
        i = atom.GetSymbol()
        if i in l:
            continue
        else:
            l.append(i)
            
bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}

SMILE_CHARSET = str(l)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

NUM_ATOMS = 76  # Maximum number of atoms
ATOM_DIM = len(l)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 1024  # Size of the latent space

smiles = data['smiles'][100]
print("SMILES:", smiles)
molecule = Chem.MolFromSmiles(smiles)
print("Num heavy atoms:", molecule.GetNumHeavyAtoms())


adjacency_tensor, feature_tensor = [], []
for smiles in data[:1000]['smiles']:
    adjacency, features = smiles_to_graph(smiles)
    adjacency_tensor.append(adjacency)
    feature_tensor.append(features)

adjacency_tensor = np.array(adjacency_tensor)
feature_tensor = np.array(feature_tensor)

print("adjacency_tensor.shape =", adjacency_tensor.shape)
print("feature_tensor.shape =", feature_tensor.shape)

wgan = GraphWGAN(generator, discriminator, discriminator_steps=1)

wgan.compile(
    optimizer_generator=keras.optimizers.Adam(5e-4),
    optimizer_discriminator=keras.optimizers.Adam(5e-4),
)

wgan.fit([adjacency_tensor, feature_tensor], epochs=10, batch_size=16)

def sample(generator, batch_size):
    z = tf.random.normal((batch_size, LATENT_DIM))
    graph = generator.predict(z)
    # obtain one-hot encoded adjacency tensor
    adjacency = tf.argmax(graph[0], axis=1)
    adjacency = tf.one_hot(adjacency, depth=BOND_DIM, axis=1)
    # Remove potential self-loops from adjacency
    adjacency = tf.linalg.set_diag(adjacency, tf.zeros(tf.shape(adjacency)[:-1]))
    # obtain one-hot encoded feature tensor
    features = tf.argmax(graph[1], axis=2)
    features = tf.one_hot(features, depth=ATOM_DIM, axis=2)
    list_molecules=[]
    for i in range(batch_size):
        list_molecules.append([adjacency[i].numpy(), features[i].numpy()])
    return list_molecules


molecules = sample(wgan.generator, batch_size=16)
graph_to_molecule((molecules[10][0],molecules[10][1]))
