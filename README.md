# KinaseDocker² - Paper reproduction code
*Code to reproduce the work described in the paper*

# Docking
A selection of activities were docked in multiple structures per kinase. Docking was done using both AutoDock VinaGPU and DiffDock.

## VinaGPU
Docking with VinaGPU required the generation of specific boxes around the binding site of each kinase structure. PyVOL was used to determine the coordinates of the boxes, following the steps in `Docking/generate_boxes.md`.

Afterwards, docking in Vina was done using VinaGPU. `Docking/dock_vina.py` was used to load the input_data (structures + smiles) and activate VinaGPU.

## DiffDock
For diffdock, two scripts were altered from the original github. `Docking/dock_diffdock.py` is the main script that is called with an input .csv file. This .csv file should contain the columns: 
- `complex_name`: unique name
- `protein_path`: path to .pdb file for kinase
- `ligand_description`: SMILES of compound
- `protein_sequence`: can be empty, since there already is a .pdb file for kinase

Furthermore the script `Docking/diffdock_utils.py` contains various helper functions.

# Database
The results of the docking steps were then used to create a database. This database contains all information about the activities, compounds, kinase structures and docked poses in the form of molfiles. This process was to a large extend manually aggregating the result files into a .sqlite database.

The next step was to use this dataset in a machine learning context.

# Input generation
In order to use the docked poses as input for a Deep Neural Network (DNN), it was needed to transform the 3D poses into machine-readable format. We decided to use PLECS fingerprints, which are molecular fingerprints that can also capture our ligand-kinase 3D interactions.

The generation of PLECS was done in `gen_plecs.py` which requires a table as input with the following information: 
- `pose_ID`: the unique ID for a pose
- `klifs_ID`: The unique ID for a kinase structure (from the KLIFS database)
- `molfile`: The actual molblock contained in a .mol file
- `pIC50`: The corresponding binding affinity

This script will then create a .npy (numpy) file and .csv file containing all plecs. In addition we decided to benchmark this against ECFP fingerprints that merely contain information about the compounds. These were generated using `gen_ecfps_all.py`

After generation of the fingerprints it was necessary to generate an input table for the DNN (ML table). These tables are generated using `create_ML_table.py` and `create_ML_table_ECFP.py` respectively. These tables contain the following information:
- `pose_ID`: unique identifier for a pose
- `accession`: UniProt kinase identifier
- `klifs_ID`: KLIFS identifier
- `InChIKey`: inchikey for compound
- `SMILES_docked`: SMILES for compound
- `pIC50`: Corresponding binding affinity
- `PLEC_index/ECFP_index`: The corresponding index of the fingerprint in the .npy file

# Machine learning (DNN)
The input files (ML_tables + fingerprints .npy) can then be used to train a DNN. The `DNN/DNN.yml` contains the conda environment in which the DNN was trained. Training and testing the DNN can be done with the `DNN/DNN.py` and `DNN/DNN_ECFP.py` scripts respectively. In addition, `DNN/datasets.py`, `DNN/datasets_ECFP.py` are needed, which handle the input data processing during DNN training and testing.

After training and testing, `DNN/create_kinase_table.py` can be used to extract all relevant results and enter that in a table. Afterwards, `DNN/create_html.py` in combination with the `kinase_table_template.html` can be used to visualise the results in a .html file.
