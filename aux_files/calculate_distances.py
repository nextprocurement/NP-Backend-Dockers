import json
import sys
from typing import List
import numpy as np
import scipy.sparse as sparse
from sparse_dot_topn import awesome_cossim_topn
import time
import pathlib
import pandas as pd

path_models_destination = pathlib.Path("data/source/cpv_models")
data_path = "data/source/place_all_embeddings_metadata_only_augmented.parquet"
topn=300
lb=0

def get_doc_by_doc_sims(W, ids_corpus) -> List[str]:
    """
    Calculates the similarity between each pair of documents in the corpus collection based on the document-topic distribution provided by the model being indexed.

    Parameters
    ----------
    W: scipy.sparse.csr_matrix
        Sparse matrix with the similarities between each pair of documents in the corpus collection.
    ids_corpus: List[str]
        List of ids of the documents in the corpus collection.

    Returns:
    --------
    sims: List[str]
        List of string representation of the top similarities between each pair of documents in the corpus collection.
    """

    # Get the non-zero elements indices
    non_zero_indices = sparse.triu(W, k=1).nonzero()

    # Convert to a string
    sim_str = \
        [' '.join([f"{ids_corpus[col]}|{W[row, col]}" for col in non_zero_indices[1]
                    [non_zero_indices[0] == row]][1:]) for row in range(W.shape[0])]

    return sim_str

for directory in path_models_destination.iterdir():
    if not directory.is_dir():
        continue
    
    t_start = time.perf_counter()
    TMfolder = pathlib.Path(directory / "model_data/TMmodel")
    thetas = sparse.load_npz(TMfolder.joinpath('thetas.npz'))
    print(f"Shape of thetas: {np.shape(thetas)} ")
    thetas_sqrt = np.sqrt(thetas)
    thetas_col = thetas_sqrt.T
    
    print(f"Topn: {topn}")
    sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)
    sparse.save_npz(TMfolder.joinpath('distances.npz'), sims)

    t_end = time.perf_counter()
    t_total = (t_end - t_start)/60
    print(f"Total computation time: {t_total}")

    corpusFile = directory.joinpath('train_data/corpus.txt')
    with corpusFile.open("r", encoding="utf-8") as f:
        lines = f.readlines()  
        f.seek(0)
        try:
            documents_ids = [line.rsplit(" 0 ")[0].strip() for line in lines]
            documents_texts = [line.rsplit(" 0 ")[1].strip().split() for line in lines]
        except:
            documents_ids = [line.rsplit("\t0\t")[0].strip() for line in lines]
            documents_texts = [line.rsplit("\t0\t")[1].strip().split() for line in lines]
    df_corpus_train = pd.DataFrame({'id': documents_ids, 'text': documents_texts})
    
    print(f"Starting similarities representation...")
    time_start = time.perf_counter()
    sim_rpr = get_doc_by_doc_sims(sims, documents_ids)
    time_end = time.perf_counter()
    print(f"Similarities representation finished in {time_end - time_start:0.4f} seconds")
    print(f"Writing similarities representation to txt file...")

    # Save similarities representation to txt file
    with open(TMfolder.joinpath('distances.txt'), 'w') as f:
        for item in sim_rpr:
            f.write("%s\n" % item)
    print(f"Saved {TMfolder.joinpath('distances.txt')}")
    print(f"Finished {directory}")
    print("-----------------------------------------------------")
    
    path_config = directory / "trainconfig.json"
    with open(path_config, "r") as f:
        config = json.load(f)
        config["TrDtSet"] = data_path
    # save the modified config
    with open(path_config, "w") as f:
        json.dump(config, f)