from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from utils.dataset_utils import load_hf_dataset, gen_pairs, gen_triplets

class TripletDataset(Dataset):
    """
    TripletDataset for contrastive training. This dataset consists of a list of (query, positive document, negative document) triplets.
    During training, the model would be optimize to produce high scores for (query, positive) pairs and low scores for (query, negative) pairs.
    Attributes
    ----------
    collection: dict 
        a dictionary that maps document id to document text
    queries: dict
        a dictionary that maps query id to query text
    triplets: list
        a list of id triplets (query_id, pos_id, neg_id)

    HINT: - make sure to implement and use the functions defined in utils/dataset_utils.py
    """
    def __init__(self, dataset_dir, split='train'):
        """
        Initialize the TripletDataset.
        Parameters
        ----------
        dataset_dir: str or Path
            path to the dataset directory
        split: str
            split of the dataset to load (train, dev, test)
        """
        assert split in ['train', 'dev', 'test'], "split must be one of ['train', 'dev', 'test']"
        
        # Load the dataset from Hugging Face Datasets
        self.dataset = load_hf_dataset(dataset_dir, split=split)
        
        # Generate triplets from the dataset
        self.triplets = gen_triplets(self.dataset)
    
    def __len__(self):
        """
        Return the number of triplets
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Get text contents of the idx-th triplet. 
        Parameters
        ----------
        idx: int 
            the index of the triplet to return
        Returns:
        tuple:
            (query_text, pos_text, neg_text) 
        """
        qid, pid, nid = self.triplets[idx]
        q_text = self.queries[qid]
        p_text = self.collection[pid]
        n_text = self.collection[nid]
        return q_text, p_text, n_text