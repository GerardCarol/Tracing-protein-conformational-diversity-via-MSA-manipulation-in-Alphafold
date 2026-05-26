import numpy as np
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import HDBSCAN
import sys
import os
import json
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
import math


parser=argparse.ArgumentParser(description='Filter and cluster proteins')
parser.add_argument('--out_dir', required=True, help='Output directory to write the centroids')
parser.add_argument('--tmscore_dictionary', required=True, help='Matrix-like dictionary file (.json) of TMScore between proteins (the indices correspond to pdb file, i.e. 1REC.pdb)')
parser.add_argument('--plddt_th', required=True, help='plddt threshold to filter the structures')
parser.add_argument('--mpnn_dir', required=True, help='Directory where to find the results of the corresponding code of protien_MPNN')
#parser.add_argument('--length_plddt_th', required=True, help='Plddt threshold for filtering disordered regions')
#parser.add_argument('--th_metapredict', required=True, help='Metapredict score threshold for filtering disordered regions')
#parser.add_argument('--length', required=True, help='Minimum percentage of the length of disordered region to consider')
#parser.add_argument('--metapredict', required=True, help='Metapredict score files directory')
parser.add_argument('--contacts', help='Write anything here if you want to filter by contacts, otherwise leave it empty')
parser.add_argument('--zscore', help='Z-score matrix file for contact filtering')
parser.add_argument('--secondary', help='Write anything here if you want to filter by secondary structure, otherwise leave it empty')
parser.add_argument('--s4pred', help='s4pred predictions file for secondary structure prediction, if you want to filter by secondary structure')
parser.add_argument('--number', required=True, help='Corresponding threshold of the zscore structures')

args=parser.parse_args()

out = args.out_dir
tmscore_file = args.tmscore_dictionary
plddt_th = float(args.plddt_th)

with open(tmscore_file, "r") as archivo:
    tmscore_all_to_all = json.load(archivo)


matrix = [list(row.values()) for row in tmscore_all_to_all.values()] #Canvia de diccionari a matriu
matrix = np.array(matrix) #Si no es fa això després dóna errors perquè els índexs creus que no són enters
ind = list(tmscore_all_to_all.keys()) #Ens diu a quina posició (i.e., "i" i "j") correspon cada pdb (nomes ho fem un cop pq es una matriu quadrada,
# això és la bijecció entre index de fila o columna i pdb)



def find_recovery_and_score(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines_start = [l for l in lines if l.startswith('>')]
    line = lines_start[-1]
    parts_line = line.split(',')
    for part in parts_line:
        elements = part.strip().split('=')
        if elements[0] == 'score':
            score = float(elements[1])
        if elements[0] == 'seq_recovery':
            seq_recovery = float(elements[1])
    return seq_recovery, score


def is_float(s):
    """Checks if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def filter_contacts(zscore, af, th):
    parser = PDBParser()
    structure = parser.get_structure("protein", af)

    model = structure[0]
    chain = model[list(model.get_chains())[0].id] # Get the first chain

    residues = list(chain.get_residues())
    num_residues = len(residues)

    # Inicializar la matriz de distancias
    distance_matrix = np.zeros((num_residues, num_residues))

    # Calcular distancias entre todos los pares de residuos
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            res1 = residues[i]
            res2 = residues[j]
            if res1.has_id('CA') and res2.has_id('CA'):
                coord1 = res1['CA'].get_coord()
                coord2 = res2['CA'].get_coord()
                distance = np.linalg.norm(coord1 - coord2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
            else:
                distance_matrix[i, j] = np.inf
                distance_matrix[j, i] = np.inf

    cmap = np.zeros((num_residues, num_residues))
    for i in range(num_residues):
        for j in range(num_residues):
            if distance_matrix[i, j] < th:  # Threshold for contacts
                cmap[i, j] = 1

    no_contact = []
    for i in range(num_residues):
        for j in range(num_residues):
            if cmap[i, j] == 1:
                good_contact = False
                for k in range(-4, 5):
                    if i + k < 0 or i + k >= num_residues:
                        continue
                    for l in range(-4, 5):
                        if j + l < 0 or j + l >= num_residues:
                                continue
                        if k**2 + l**2 <= 16:
                            if zscore[i+k, j+l] > 1:
                                good_contact = True
                                break
                    if good_contact:
                        break
                if not good_contact:
                    no_contact.append((i, j))
    
    data = np.array(no_contact)
    if len(data) > 0:
        if len(data) <= 4:
            return True
        hdbscan = HDBSCAN(min_cluster_size=4, cluster_selection_epsilon=4)
        labels = hdbscan.fit_predict(data)
        unique_labels = set(labels)
        for lab in unique_labels:
            if lab != -1:
                len_lab = np.sum(labels == lab)
                if len_lab >= 10:
                    return False
    return True






def filter_secondary_structure(s4pred_file, pdb_file, file_to_tmp, out_dssp):
    data_secondary = []
    with open(s4pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment/header lines starting with '#' 
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 6:  # Expecting 6 columns: index, AA, pred_SS, P(C), P(H), P(E) 
                try:
                    residue_number = int(parts[0])
                    amino_acid = parts[1]
                    predicted_ss = parts[2]
                    prob_coil = float(parts[3])
                    prob_helix = float(parts[4])
                    prob_strand = float(parts[5])
                    data_secondary.append([
                        residue_number,
                        amino_acid,
                        predicted_ss,
                        prob_coil,
                        prob_helix,
                        prob_strand
                    ])
                except ValueError as e:
                    print(f"Warning: Could not parse line: '{line}' - Error: {e}")
                    continue
            else:
                print(f"Warning: Skipping line due to unexpected column count: '{line}'")
    
    helices = []
    betas = []
    for row in data_secondary:
        if row[2] == 'H' and row[4] >= 0.85:  # Check if the predicted secondary structure is helix and probability is high.   ABANS 0.95
            helices.append(row[0])
        elif row[2] == 'E' and row[5] >= 0.85:  # Check if the predicted secondary structure is strand and probability is high.    ABANS 0.95
            betas.append(row[0])

    pairs_helices = []
    if helices:
        helices.sort()  # Sort the helices to ensure they are in order
        first_helix = helices[0]
        for i in range(len(helices)):
            if i+1 == len(helices):
                pairs_helices.append((first_helix, helices[i]))
                break
            elif helices[i]+1 != helices[i+1]:
                pairs_helices.append((first_helix, helices[i]))
                first_helix = helices[i+1]
    else:
        pairs_helices = None

    pairs_betas = []
    if betas:
        betas.sort()  # Sort the beta strands to ensure they are in order
        first_beta = betas[0]
        for i in range(len(betas)):
            if i+1 == len(betas):
                pairs_betas.append((first_beta, betas[i]))
                break
            elif betas[i]+1 not in betas:
                pairs_betas.append((first_beta, betas[i]))
                first_beta = betas[i+1]
    else:
        pairs_betas = None
    

    parser = PDBParser(PERMISSIVE=True)
    structure = parser.get_structure("protein", pdb_file)
    io = PDBIO()
    io.set_structure(structure)

    with open(file_to_tmp, 'w') as f:
        f.write("HEADER\n") # Encabezado mínimo
        io.save(f)

    #Identificar la cadena dinámicamente para evitar el KeyError
    # El objeto 'structure' tiene un diccionario con los modelos. En este caso, solo hay uno.
    model = structure[0]
    dssp = DSSP(model, file_to_tmp, dssp="mkdssp")
    os.remove(file_to_tmp)  # Remove the temporary file after DSSP processing

    alpha_helices_resids = []
    beta_strands_resids = []
    with open(out_dssp, 'w') as f:
        # Escribir la cabecera
        f.write("Residuo\tEstructura Secundaria\n")
        f.write("-------\t---------------------\n")
        
        # Recorrer los resultados de DSSP y escribir en el archivo
        # La salida de DSSP contiene:
        # (res_id, aminoácido, sec_struc, accesibilidad, ...)
        for dssp_data in dssp:
            # dssp_data[0] es la tupla del identificador del residuo
            # dssp_data[1] es el aminoácido
            # dssp_data[2] es la estructura secundaria
            res_id = dssp_data[0]
            amino_acid = dssp_data[1]
            secondary_structure = dssp_data[2]

            if secondary_structure == 'H' or secondary_structure == 'G' or secondary_structure == 'I':
                alpha_helices_resids.append(res_id)
            
            if secondary_structure == 'E' or secondary_structure == 'B':
                beta_strands_resids.append(res_id)
            
            # Formatear la línea de salida
            line = f"{amino_acid}\t{res_id}\t{secondary_structure}\n"
            f.write(line)
        
        f.write("\nResumen de Estructura Secundaria:\n")
        f.write(f"Número de hélices: {len(alpha_helices_resids)}\n")
        f.write(f"Número de beta strands: {len(beta_strands_resids)}\n")


    '''
    for key in dssp.keys():
        # key es una tupla (chain_id, (hetflag, resseq, icode))
        chain_id = key[0] # Aunque solo haya una cadena, este será su ID (ej. 'A')
        res_id_tuple = key[1] # (hetflag, resseq, icode)
        res_num = res_id_tuple[1] # El número de secuencia del residuo (resseq)
        # El código de estructura secundaria está en el índice 2 de la tupla de datos de DSSP
        ss_code = dssp[key][2]

        if ss_code == 'H' or ss_code == 'G' or ss_code == 'I':
            alpha_helices_resids.append(res_num)
        if ss_code == 'E' or ss_code == 'B':
            beta_strands_resids.append(res_num)
    '''

    pairs_helices_dssp = []
    if alpha_helices_resids:
        initial_res = alpha_helices_resids[0] if alpha_helices_resids else None
        for i in range(len(alpha_helices_resids)):
            if  alpha_helices_resids[i] + 1 not in alpha_helices_resids:
                pairs_helices_dssp.append((initial_res, alpha_helices_resids[i]))
                if i + 1 < len(alpha_helices_resids):
                    initial_res = alpha_helices_resids[i + 1]
    else:
        pairs_helices_dssp = None

    pairs_betas_dssp = []
    if beta_strands_resids: 
        initial_res = beta_strands_resids[0] if beta_strands_resids else None
        for i in range(len(beta_strands_resids)):# If no significant overlap, return True to keep the protei
            if beta_strands_resids[i] + 1 not in beta_strands_resids:
                pairs_betas_dssp.append((initial_res, beta_strands_resids[i]))
                if i + 1 < len(beta_strands_resids):
                    initial_res = beta_strands_resids[i + 1]
    else:
        pairs_betas_dssp = None
    

    not_recovered_helices = 0
    file_intervals = out_dssp.replace('.txt', '_helices.txt')
    helices_counter = 0
    if pairs_helices is not None:
        for pair in pairs_helices:
            length_pair = pair[1] - pair[0] + 1
            if length_pair >= 5:
                best_overlap = 0
                helices_counter += 1
                if pairs_helices_dssp is not None:
                    for pair_dssp in pairs_helices_dssp:
                        overlap = max(0, min(pair[1], pair_dssp[1]) - max(pair[0], pair_dssp[0]) + 1)
                        if overlap > best_overlap:
                            best_overlap = overlap

                    if length_pair >= 10:
                        if best_overlap/length_pair <= 0.8:  # Abans 0.75
                            not_recovered_helices += 1
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) is NOT recovered\n")
                        else:
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) IS recovered\n")
                    else:
                        if best_overlap < length_pair - 1:
                            not_recovered_helices += 1
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) is NOT recovered\n")
                        else:
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) IS recovered\n")
    



    not_recovered_betas = 0
    file_intervals = out_dssp.replace('_dssp.txt', '_betas.txt')
    betas_counter = 0
    if pairs_betas is not None:
        for pair in pairs_betas:
            length_pair = pair[1] - pair[0] + 1
            if length_pair >= 4:
                best_overlap = 0
                betas_counter += 1
                if pairs_betas_dssp is not None:
                    for pair_dssp in pairs_betas_dssp:
                        overlap = max(0, min(pair[1], pair_dssp[1]) - max(pair[0], pair_dssp[0]) + 1)
                        if overlap > best_overlap:
                            best_overlap = overlap

                    if length_pair >= 10:
                        if best_overlap/length_pair <= 0.8:  # Abans 0.75
                            not_recovered_betas += 1
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) is NOT recovered\n")
                        else:
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) IS recovered\n")
                    else:
                        if best_overlap < length_pair - 1:
                            not_recovered_betas += 1
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) is NOT recovered\n")
                        else:
                            with open(file_intervals, 'a') as f:
                                f.write(f"({pair[0]}, {pair[1]}) IS recovered\n")
    
    # Aquestes línies són noves, per no tenir problemes si no s'han trobat helices o betas
    if pairs_helices_dssp is None:
        if pairs_helices is not None:
            not_recovered_helices = len(pairs_helices)
            helices_counter = len(pairs_helices)
        else:
            not_recovered_helices = 0
            helices_counter = 0
    if pairs_betas_dssp is None:
        if pairs_betas is not None:
            not_recovered_betas = len(pairs_betas)
            betas_counter = len(pairs_betas)
        else:
            not_recovered_betas = 0
            betas_counter = 0



    frac_not_recovered_helices = not_recovered_helices / helices_counter if helices_counter!=0 else 0
    frac_not_recovered_betas = not_recovered_betas / betas_counter if betas_counter!=0 else 0




    with open(file_intervals, 'a') as f:
        f.write(f"Fraction of helices not recovered: {frac_not_recovered_helices:.2f}\n")
        f.write(f"Fraction of beta strands not recovered: {frac_not_recovered_betas:.2f}\n")


    chain_id = list(model.get_chains())[0].id # Obtener el ID de la primera cadena
    chain = model[chain_id]
    num_residuos = len(list(chain.get_residues()))
    num_of_recovery = num_residuos/100
    # Arrodonim a l'enter més proper::
    if num_of_recovery - math.floor(num_of_recovery) < 0.5:
        num_of_recovery = math.floor(num_of_recovery)
    else:
        num_of_recovery = math.ceil(num_of_recovery)

    #if frac_not_recovered_helices >= 0.2 or frac_not_recovered_betas >= 0.2 or not_recovered_helices >= 3 or not_recovered_betas >= 3:
    if frac_not_recovered_helices >= 0.2 or frac_not_recovered_betas >= 0.2 or not_recovered_helices > num_of_recovery or not_recovered_betas > num_of_recovery:
        print(f"The protein {pdb_file} has a significant number of helices or beta strands not recovered by DSSP. The protein will be removed from the list.\n")
        return False
    else:
        print(f"The protein {pdb_file} has a significant number of helices and beta strands recovered by DSSP. The protein will be kept in the list.\n")
        return True 
        



                
'''
metapredict_file = args.metapredict
with open(metapredict_file, 'r') as m:
    lines = m.readlines()
line = lines[0].strip()
parts = line.split(',')

metapredict_scores = []
bool_float = False
for part in parts:
    part = part.strip()
    if is_float(part) and not bool_float:
        bool_float = True
    if bool_float:
        metapredict_scores.append(float(part))
'''

#
'''
good_plddt_indices = []
for i, protein in enumerate(ind):
    plddt_values = []
    file_cluster = protein
    with open(file_cluster, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                plddt = float(line[61:66].strip())
                plddt_values.append(plddt)
    media_plddt = sum(plddt_values) / len(plddt_values)
    if media_plddt >= plddt_th: # and qmean_scores[j]>=th:
        good_plddt_indices.append(i)
'''

plddts = []
good_plddt_indices = []
#good_pdbs = []
for i, pdb in enumerate(ind):
    pdb = pdb.strip()
    if pdb.startswith("/orozco/projects"):
        pdb = pdb.replace("/orozco/projects", "/data/mmb", 1)
    
    with open(pdb, 'r') as p:
        pdb_lines = p.readlines()

    residues = {}
    for line in pdb_lines:
        if line.startswith("ATOM"):
            residu = int(line[23:26].strip())
            if residu not in residues:
                plddt = float(line[61:66].strip())
                residues[residu] = plddt
    
    media_plddt = sum(residues.values()) / len(residues)
    
    ##############################################################################################################################################
    ##############################################################################################################################################
    # Just use plddt, not metapredict
    if media_plddt >= plddt_th:
        print(f"{pdb} has a mean pLDDT of {media_plddt:.2f} above the threshold {plddt_th}. The protein will be kept in the list.\n")
        good_plddt_indices.append(i)
        plddts.append(media_plddt)
    ###############################################################################################################################################
    ###############################################################################################################################################

    '''
    if media_plddt >= plddt_th:  # Filter based on pLDDT threshold
        plddts.append(media_plddt)  # Store the pLDDT value for each protein
        name = os.path.basename(pdb)
        name = name.split('.')[0]
        metapredict_file = args.metapredict

        with open(metapredict_file, 'r') as m:
            lines = m.readlines()
        line = lines[0].strip()
        parts = line.split(',')
        
        
        if len(metapredict_scores) != len(residues):
            ValueError(f"Length of metapredict scores ({len(metapredict_scores)}) does not match number of residues ({len(residues)}) in {pdb}\n")
        
        disordered_region = []
        count = 0
        good = True
        
        for residue in residues:
            if residues[residue] < float(args.length_plddt_th):
                disordered_region.append(residue)
                count = 0
            else:
                count = count + 1
            
            if count >= 3:
                longitude = len(disordered_region)/len(residues)

                if longitude >= float(args.length):
                    print(f"{pdb} has a disordered region of length {len(disordered_region)}")
                    mean_metapredict = sum(metapredict_scores[residue - 1] for residue in disordered_region) / len(disordered_region)
                    print(f"Mean metapredict score for disordered region: {mean_metapredict:.2f}\n")

                    if mean_metapredict < float(args.th_metapredict):
                        print(f"The protein {pdb} has a disordered region with a mean metapredict score of {mean_metapredict:.2f} below the threshold {args.th_metapredict}. The protein will be removed from the list.\n")
                        good = False
                        break
                    else:
                        disordered_region = []
                        count = 0
                else:
                    disordered_region = []
                    count = 0

        if good:
            #good_pdbs.append(pdb)
            good_plddt_indices.append(i)
            print(f"{pdb} is a good pdb with no long disordered regions or with acceptable metapredict scores.")
    '''



if args.contacts:
    filtered_contacts = []
    if not args.zscore:
        print("You need to provide a zscore file for contact filtering when using the --contacts option.")
        sys.exit(1)
    zscore = np.loadtxt(args.zscore, delimiter=' ')
    for i in good_plddt_indices:
        af = ind[i]
        if af.startswith("/orozco/projects"):
            af = af.replace("/orozco/projects", "/data/mmb", 1)
        if filter_contacts(zscore, af, th=8): # El th és el de la distància per considerar un contacte
            filtered_contacts.append(i)
    
    good_plddt_indices = filtered_contacts





if args.secondary:
    if not args.s4pred:
        print("You need to provide an S4Pred file for secondary structure filtering when using the --secondary option.")
        sys.exit(1)
    filtered_secondary = []

    # Generate directory in the filters directory for the th of the zscore:
    dir_filters = os.path.dirname(args.s4pred)
    dir_filters_number = os.path.join(dir_filters, f"th_{args.number}")
    if not os.path.exists(dir_filters_number):
        os.makedirs(dir_filters_number)

    for i in good_plddt_indices:
        af = ind[i]
        if af.startswith("/orozco/projects"):
            af = af.replace("/orozco/projects", "/data/mmb", 1)
        
        file_to_tmp = os.path.join(dir_filters_number, os.path.basename(af))
        out_dssp = os.path.join(dir_filters_number, os.path.basename(af).replace('.pdb', '_dssp.txt'))

        while os.path.exists(file_to_tmp):  # Ensure the temporary file does not already exist
            file_to_tmp = file_to_tmp.replace('.pdb', '_tmp.pdb')
        while os.path.exists(out_dssp):  # Ensure the dssp output file does not already exist
            out_dssp = out_dssp.replace('_dssp.txt', '_dssp_tmp.txt')

        if filter_secondary_structure(args.s4pred, af, file_to_tmp, out_dssp):
            filtered_secondary.append(i)
            with open(f'{dir_filters_number}/filtered_secondary.txt', 'a') as f:
                f.write(af + '\n')
        else:
            with open(f'{dir_filters_number}/filtered_secondary.txt', 'a') as f:
                f.write(f"{af} has a significant number of helices or beta strands not recovered by DSSP. The protein will be removed from the list.\n")

    good_plddt_indices = filtered_secondary
    



# Here, filter of seq_recovery and score from protein_MPNN
score_ths = np.around(np.arange(0.8, 1.11, 0.05), decimals=2)
seq_recovery_ths = np.around(np.arange(0.2, 0.351, 0.05), decimals=2)

copy_original_good_plddt_indices = good_plddt_indices.copy()
for score_th in score_ths:
    for seq_recovery_th in seq_recovery_ths:

        good_plddt_indices = []
        for i in copy_original_good_plddt_indices:
            af = ind[i]
            basename_mpnn = f'{os.path.splitext(os.path.basename(af))[0]}.fa'
            file_try = os.path.join(os.path.join(os.path.join(os.path.join(args.mpnn_dir, 'first'), args.number), 'seqs'), basename_mpnn)
            
            if not os.path.exists(file_try):
                file_try = os.path.join(os.path.join(os.path.join(os.path.join(args.mpnn_dir, 'second'), args.number), 'seqs'), basename_mpnn)
            seq_recovery, score = find_recovery_and_score(file_try)

            if seq_recovery >= seq_recovery_th and score <= score_th:
               good_plddt_indices.append(i)

        good_plddt_indices = list(set(good_plddt_indices))





        if len(good_plddt_indices) >= 2:
            submatrix = matrix[good_plddt_indices, :][:, good_plddt_indices]
            # cluster_th = 3.5
            #------------------------------------------------------------ATENCIÓ--------------------------------------------------------------
            #CANVIAR EL TH AL QUE VULGUEM, ABANS ERA 6
            cluster_th = 0.70
            submatrix = 1 - submatrix # Convertir TMScore a una "dis-similarity" per al clustering
            clusters_agg = AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_th, compute_full_tree=True, metric='precomputed', linkage='complete').fit_predict(submatrix) 

            # Find the centroids
            unique_labels = np.unique(clusters_agg)
            '''
            centroids = []
            for label in unique_labels:
                cluster_indices = np.where(clusters_agg == label)[0]
                centroid = np.mean(matrix[cluster_indices], axis=0)
                centroids.append(centroid)

            # Find the nearest protein to each centroid
            nearest_proteins = []
            for centroid in centroids:
                distances_to_centroid = np.linalg.norm(matrix - centroid, axis=1)
                nearest_idx = np.argmin(distances_to_centroid)
                nearest_protein = nearest_idx  # This should be your protein index in your dataset
                nearest_proteins.append(nearest_protein)'
            '''


            #----------------------------------------------------ATENCIÓ-------------------------------------------------
            # Per triar el centroid, podríem provar de fer-ho agafant el que té un major plddt
            #------------------------------------------------------------------------------------------------------------

            nearest_proteins = []
            for label in unique_labels:
                # Get the indices of the proteins in the current cluster
                cluster_indices = np.where(clusters_agg == label)[0]
                
                # Calculate the average pairwise distance for each protein in the cluster
                cluster_distances = matrix[cluster_indices][:, cluster_indices]  # Submatrix of distances between proteins in the same cluster
                
                # The centroid is the average pairwise distance (mean of the submatrix)
                centroid = np.mean(cluster_distances, axis=1)  # Mean of each row (average distance to others in the cluster)

                # Step 3: Find the protein closest to the centroid of each cluster
                # To store the closest protein index for each cluster
                if len(centroid) == 1:
                    nearest_proteins.append(cluster_indices[0])  # The only protein in the cluster is the nearest
                    continue
                # Calculate the distance of each protein in the cluster to the centroid
                
                # Find the index of the protein closest to the centroid (smallest distance)
                #nearest_protein_index = np.argmin(centroid)
                
                # Save the index of the nearest protein to the centroid
                #nearest_proteins.append(cluster_indices[nearest_protein_index])
                

                # AQUÍ FEM EL CENTROID QUE SIGUI EL QUE TÉ UN MAJOR pLDDT
                plddt_values = [plddts[i] for i in cluster_indices]  # Get pLDDT values for proteins in the cluster
                max_plddt_index = np.argmax(plddt_values)  # Index of the protein with the highest pLDDT in the cluster
                nearest_proteins.append(cluster_indices[max_plddt_index])  # Append the index of the protein with the highest pLDDT


            for protein in nearest_proteins:
                full_path = ind[good_plddt_indices[protein]]
                with open(os.path.join(out, f'centroids_score_{score_th}_recovery_{seq_recovery_th}.txt'), 'a') as f:
                #with open(os.path.join(out, f'centroids_only_plddt.txt'), 'a') as f:
                    f.write(full_path + '\n')
        else:
            if len(good_plddt_indices) == 1:
                full_path = ind[good_plddt_indices[0]]
                with open(os.path.join(out, f'centroids_score_{score_th}_recovery_{seq_recovery_th}.txt'), 'a') as f:
                #with open(os.path.join(out, f'centroids_only_plddt.txt'), 'a') as f:
                    f.write(full_path + '\n')
            #else:
            #    print("No proteins with sufficient pLDDT or no clusters found. Exiting.")
            #    sys.exit(0)

