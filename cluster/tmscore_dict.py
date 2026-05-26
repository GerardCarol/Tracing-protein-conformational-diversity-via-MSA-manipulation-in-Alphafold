import argparse
import os
import json
import subprocess
import re

parser=argparse.ArgumentParser(description='Create TMScore dictionary of distance (TMScore) between experimental conformations and alphafold generated conformations.')
parser.add_argument('--dir_af_1', required=True, help='Path of the alphafold structures directory (first)')
parser.add_argument('--dir_af_2', required=True, help='Path of the alphafold structures directory (second)')
parser.add_argument('--pdb_code', required=True, help='PDB code of the experimental proteins (the sequence of which is being used for AF)')
parser.add_argument('--tmscore_file', required=True, help='Output file for the TMScore (.json)')
args=parser.parse_args()

dir_1 = args.dir_af_1
dir_2 = args.dir_af_2
tmscore_file = args.tmscore_file

pdbs_1 = [os.path.join(dir_1, nombre) for nombre in os.listdir(dir_1)]
pdbs_2 = [os.path.join(dir_2, nombre) for nombre in os.listdir(dir_2)]
pdbs = pdbs_1 + pdbs_2

tmscore_all_to_all={}
check = {pdb1: {pdb2: False for pdb2 in pdbs} for pdb1 in pdbs}
for pdb1 in pdbs:
    tmscore_all_to_all[pdb1]={}
    for pdb2 in pdbs:
            cmd = ['/data/mmb/TransAtlas/Gerard/MontseScripts/transatlas_af/prova/TMscore', '-seq', pdb1, pdb2]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Toda la salida como string
            output = result.stdout

            # Mostrar la salida completa (opcional)
            print(output)

            # Ejemplo: extraer el TM-score con regex
            tm_score_match = re.search(r'TM-score\s*=\s*([0-9.]+)', output)
            if tm_score_match:
                tm_score = float(tm_score_match.group(1))
                print(f"TM-score: {tm_score}")
                tmscore_all_to_all[pdb1][pdb2] = tm_score
            else:
                print("No se encontró TM-score en la salida. Assuming it is 0")
                tmscore_all_to_all[pdb1][pdb2] = 0


with open(tmscore_file, "w") as archivo:
    json.dump(tmscore_all_to_all, archivo)
