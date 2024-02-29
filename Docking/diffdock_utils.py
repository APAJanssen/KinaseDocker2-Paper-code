'''
Edited utils file with helper functions for diffdock docking.
'''
import os

import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit import Chem # Extra import
from rdkit.Geometry import Point3D # Extra import
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm

from datasets.process_mols import parse_pdb_from_path, generate_conformer, read_molecule, get_lig_graph_with_matching, \
    extract_receptor_structure, get_rec_graph

# Extra imports
import pandas as pd
import numpy as np
import copy
import zlib

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence


def set_nones(l):
    return [s if str(s) != 'nan' else None for s in l]


def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile(protein_files[i]))
        else:
            new_sequences.append(protein_sequences[i])
    return new_sequences

# Extra Function
def get_sequence(protein_file, protein_sequence):
    if protein_file is not None:
        new_sequence = get_sequences_from_pdbfile(protein_file)
    else:
        new_sequence = protein_sequence

    return new_sequence


def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    return embeddings


def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None

# Extra Function
def preprocess_data(filepath):
    '''
    Expects .csv of the form: complex_name (md5 hash of klifs + smiles), protein_path, ligand_description, protein_sequence
    '''
    data = pd.read_csv(filepath)
    
    # EXTRA CLAUSE: Remove every smiles + klifs combo that is already in the results
    output_folder = f'results/{filepath.split("/")[-1].split(".")[0]}'

    if os.path.exists(output_folder):
        dfs = []
        
        for klifs in data['protein_path'].apply(lambda x: x.split('/')[-1].split('.')[0]).unique():
            try:
                dfs.append(pd.read_csv(os.path.join(output_folder, f'results_{klifs}.csv')))
            except:
                pass
        
        if len(dfs):
            results = pd.concat(dfs)
            results['klifs_smiles'] = results['klifs_ID'].apply(int).apply(str) + results['SMILES_input']

            data['klifs_smiles'] = data['protein_path'].apply(lambda x: x.split('/')[-1].split('.')[0]) + data['complex_name']
            data = data[~data['klifs_smiles'].isin(results['klifs_smiles'])]
            data.drop(columns=['klifs_smiles'], inplace=True)

            print('# Of SMILES to dock:', len(data))

            if len(data) == 0:
                print('EVERY SMILES WAS SUCCESFULLY DOCKED')
                exit()

    data['klifs'] = data['protein_path'].apply(lambda x: int(x.split('/')[-1].split('.')[0])) # Expects protein input files to be {klifs}.pdb
    grouped = data.groupby(['klifs'])

    chunks = [(klifs, grouped.get_group(klifs)) for klifs in sorted(list(set(data['klifs'])))]
    return chunks


# Extra function
def postprocess_data(lig, ligand_pos, klifs, smiles_input, confidence_list, remove_hs):
    results = {'klifs_ID': [], 'SMILES_input': [], 'SMILES_output': [], 'molfile_compressed': [], 'DiffDock_confidence': []}

    for rank, pos in enumerate(ligand_pos):
        mol_pred = copy.deepcopy(lig)
        if remove_hs: mol_pred = Chem.RemoveHs(mol_pred)

        conf = mol_pred.GetConformer()
        for i in range(mol_pred.GetNumAtoms()):
            x,y,z = pos.astype(np.double)[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))

        smiles_output = Chem.MolToSmiles(mol_pred)
        molfile = Chem.MolToMolBlock(mol_pred)
        confidence = confidence_list[rank]

        # Fix molfile header and gzip it
        header = smiles_input + f'_{str(klifs)}_DiffDock_{rank + 1}\n'
        index = molfile.index('3D') + 2
        molfile = header + molfile[index:]
        molfile_compressed = compress_string(molfile)

        results['klifs_ID'].append(int(klifs))
        results['SMILES_input'].append(smiles_input)
        results['SMILES_output'].append(smiles_output)
        results['molfile_compressed'].append(molfile_compressed)
        results['DiffDock_confidence'].append(confidence)

    return pd.DataFrame(results)

# Extra function
def compress_string(string):
    """
    Compresses a string
    Arguments:
        string (str)              : string to compress  
    Returns:
        compressed (str)          : compressed string
    """ 
    return zlib.compress(string.encode('utf-8')).hex()


class InferenceDataset(Dataset):
    def __init__(self, out_dir, complex_names, protein_files, ligand_descriptions, protein_sequences, lm_embedding,
                 receptor_radius=30, c_alpha_max_neighbors=None, precomputed_lm_embedding=None,
                 remove_hs=False, all_atoms=False, atom_radius=5, atom_max_neighbors=None):

        super(InferenceDataset, self).__init__()
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors

        self.complex_names = complex_names
        self.protein_files = protein_files
        self.protein_file = protein_files[0] # Extra, since it's the same one over and over again
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences
        self.protein_sequence = protein_sequences[0]

        # Generate a single LM embedding - Extra
        if lm_embedding and (precomputed_lm_embedding is None):
            print("Generating ESM language model embedding")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequence = get_sequence(self.protein_file, self.protein_sequence)
            sequences = protein_sequence.split(':')
            labels = [self.protein_file + '_chain_' + str(j) for j in range(len(sequences))]

            lm_embedding = compute_ESM_embeddings(model, alphabet, labels, sequences)
            self.lm_embedding = list(lm_embedding.values())
        elif not lm_embedding:
            self.lm_embedding = [None]
        else:
            self.lm_embedding = precomputed_lm_embedding

        # generate LM embeddings
        # if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
        #     print("Generating ESM language model embeddings")
        #     model_location = "esm2_t33_650M_UR50D"
        #     model, alphabet = pretrained.load_model_and_alphabet(model_location)
        #     model.eval()
        #     if torch.cuda.is_available():
        #         model = model.cuda()

        #     protein_sequences = get_sequences(protein_files, protein_sequences)
        #     labels, sequences = [], []
        #     for i in range(len(protein_sequences)):
        #         s = protein_sequences[i].split(':')
        #         sequences.extend(s)
        #         labels.extend([complex_names[i] + '_chain_' + str(j) for j in range(len(s))])

        #     print(labels, sequences)
        #     print(len(labels), len(sequences))

        #     lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

        #     self.lm_embeddings = []
        #     for i in range(len(protein_sequences)):
        #         s = protein_sequences[i].split(':')
        #         self.lm_embeddings.append([lm_embeddings[complex_names[i] + '_chain_' + str(j)] for j in range(len(s))])

        # elif not lm_embeddings:
        #     self.lm_embeddings = [None] * len(self.complex_names)

        # else:
        #     self.lm_embeddings = precomputed_lm_embeddings

        # generate structures with ESMFold
        if None in protein_files:
            print("generating missing structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()

            for i in range(len(protein_files)):
                if protein_files[i] is None:
                    self.protein_files[i] = f"{out_dir}/{complex_names[i]}/{complex_names[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(model, self.protein_files[i], protein_sequences[i])

    def len(self):
        return len(self.complex_names)

    def get(self, idx):
        name, protein_file, ligand_description, lm_embedding = \
            self.complex_names[idx], self.protein_file, self.ligand_descriptions[idx], self.lm_embedding

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph['name'] = name # == SMILES

        # parse the ligand, either from file or smile
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path

            if mol is not None:
                mol = AddHs(mol)
                generate_conformer(mol, maxAttempts=100)
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception('RDKit could not read the molecule ', ligand_description)
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol, maxAttempts=100)
        except Exception as e:
            print('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
            complex_graph['success'] = False
            return complex_graph

        try:
            # parse the receptor from the pdb file
            rec_model = parse_pdb_from_path(protein_file)
            get_lig_graph_with_matching(mol, complex_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
                                        num_conformers=1, remove_hs=self.remove_hs)
            rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(rec_model, mol, lm_embedding_chains=lm_embedding)
            if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                complex_graph['success'] = False
                return complex_graph

            get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius=self.receptor_radius,
                          c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                          atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            complex_graph['success'] = False
            return complex_graph

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['ligand'].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = mol
        complex_graph['success'] = True
        return complex_graph # extra return the smiles in ['name'] attribute
