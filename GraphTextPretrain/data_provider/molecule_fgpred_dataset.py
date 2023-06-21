import torch
from torch_geometric.data import Dataset
import os
import numpy as np
from rdkit import Chem
from descriptastorus.descriptors import rdDescriptors

RDKIT_PROPS = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
               'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
               'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
               'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
               'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
               'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
               'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
               'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
               'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
               'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
               'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
               'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
               'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
               'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

# def rdkit_functional_group_label_features_generator(mol):
#     """
#     Generates functional group label for a molecule using RDKit.

#     :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
#     :return: A 1D numpy array containing the RDKit 2D features.
#     """
#     smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
#     generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
#     features = generator.process(smiles)[1:]
#     features = np.array(features)
#     features[features != 0] = 1
#     return features

def rdkit_functional_group_label_features_generator(smiles):
    """
    Generates functional group label for a molecule using RDKit.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    mol = Chem.MolFromSmiles(smiles)
    smiles2 = Chem.MolToSmiles(mol, isomericSmiles=True)
    # assert smiles == smiles2, print(smiles, smiles2)
    generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
    features = generator.process(smiles2)[1:]
    features = np.array(features)
    features[features != 0] = 1
    features = features * 2 - 1
    return features


def rdkit_functional_group_label_features_generator_regression(smiles):
    """
    Generates functional group label for a molecule using RDKit.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing the RDKit 2D features.
    """
    mol = Chem.MolFromSmiles(smiles)
    smiles2 = Chem.MolToSmiles(mol, isomericSmiles=True)
    # assert smiles == smiles2, print(smiles, smiles2)
    generator = rdDescriptors.RDKit2D(RDKIT_PROPS)
    features = generator.process(smiles2)[1:]
    features = np.array(features)
    return features


class MoleculeFGPred(Dataset):
    def __init__(self, root, text_max_len, prompt=None, task_type='classfication'):
        super(MoleculeFGPred, self).__init__(root)
        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root+'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = None
        
        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt
        self.num_labels = len(RDKIT_PROPS)
        self.task_type = task_type

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        smiles_name = self.smiles_name_list[index]

        # load and process graph
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        # load and process smiles
        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        with open(smiles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            smiles = lines[0].strip()

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        
        if self.task_type == 'classification':
            fg_feat = rdkit_functional_group_label_features_generator(smiles)
        elif self.task_type == 'regression':
            fg_feat = rdkit_functional_group_label_features_generator_regression(smiles)
        else:
            raise NotImplementedError()
        fg_feat = torch.from_numpy(fg_feat).float()
        return data_graph, smiles_prompt, fg_feat
    