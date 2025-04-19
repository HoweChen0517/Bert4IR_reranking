from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from utils.dataset_utils import read_nway_data
import json
from ir_measures import read_trec_run
from collections import Counter

class NwayDataset(Dataset):
    def __init__(
        self, collection_path: str, queries_path: str, train_nway_path: str, n_way:int=64
    ):
        """
        Constructing a NwayDataset
        Parameters
        ----------
        collection_path: str
            path to a tsv file,  each line has a document id and document text separated by a tab character
        queries_path: str
            path to a tsv file,  each line has a query id and query text separated by a tab character
        train_nway_path: str
            path to a tsv file,  each line has a query id, doc_id, rank, score and label separated by a tab character
        """
        self.collection = {}
        with open(collection_path, "r", encoding='utf-8') as f:
            for line in f:
                id_, text = line.rstrip("\n").split("\t")
                self.collection[id_] = text
                
        self.queries = {}
        with open(queries_path, "r", encoding='utf-8') as f:
            for line in f:
                id_, text = line.rstrip("\n").split("\t")
                self.queries[id_] = text
                
        self.n_way = n_way
                
        self.nway_data = read_nway_data(train_nway_path, n_way=self.n_way)
        
    def __len__(self):
        """
        Return the number of nway_data
        """
        return len(self.nway_data)
        
    def __getitem__(self, idx):
        """
        Get text contents of the idx-th nway_data. 
        Parameters
        ----------
        idx: int 
            the index of the triplet to return
        Returns:
        tuple:
            (query_text, (nway docs), (nway labels)) 
        """
        qid, dids, labels = self.nway_data[idx]
        q_text = self.queries[qid]
        d_texts = [self.collection[did] for did in dids]
        labels = [int(label) for label in labels]
        
        return q_text, d_texts, labels

    