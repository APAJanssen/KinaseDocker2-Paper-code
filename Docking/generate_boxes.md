# Determine KLIFS pockets:
1. Get [PyVol](https://github.com/schlessinger-lab/pyvol) plugin
2. Get template config file
```bash
pyvol -t template.cfg
```
3. Create script that alters this config file to generate pocket per KLIFS
```bash
prot_file = {path to protein.pdb}
project_dir = {output directory}
prefix = {filename}
protein_only = True
display_mode = spheres
protein = {protein_name} # maybe unimportant
```
4. Run pyVol
```bash
python -m pyvol <input_parameters.cfg>
```
## Failed:
- Manually pick best in pymol with PyVol plugin:
	- Mode = all
	- Output_dir can be specified where all .xyzrg files will end up to pick one

# Determine VINA boxes:
1. Read out generated .xyzrg files and determine center and box size
2. Write boxes to file and use in vinaGPU
