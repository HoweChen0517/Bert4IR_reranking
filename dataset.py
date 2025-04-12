import torch
from torch.utils.data import Dataset, DataLoader
import os
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def gen_triplets(dataset):
    """
    生成MS MARCO数据集的三元组
    Return: triplets: [(query, passage, label), ...]
    """

    triplets = []

    for example in tqdm(dataset, total=len(dataset), desc='Processing triplets'):
        query = example["query"]
        passages = example["passages"]["passage_text"]
        labels = example["passages"]["is_selected"]
        for passage, label in zip(passages, labels):
            triplets.append((query, passage, label))

    #print(f'For current dataset, # queries={len(dataset)}, # triplets_records={len(triplets)}')

    return triplets

def gen_duples(dataset):
  """
  生成MS MARCO数据集的多元组
  Return: duples: [(query, pos_doc, neg_doc), ...]
  """

  duples = []

  for example in tqdm(dataset, total=len(dataset), desc='Processing duples'):
      query = example["query"]
      passages = example["passages"]["passage_text"]
      labels = example["passages"]["is_selected"]

      pos_docs = [p for p, l in zip(passages, labels) if l == 1]
      neg_docs = [p for p, l in zip(passages, labels) if l == 0]

      # 构造正负配对（可采样多个 negative）
      for pos_doc in pos_docs:
          for neg_doc in neg_docs:
              duples.append((query, pos_doc, neg_doc))

  return duples

class TripletDataset(Dataset):
  """
  构造三元组dataset: e.g. <query, doc, label>
  将query-document按pointwise的形式拼在一起送入bert
  """
  def __init__(self,
               queries,                 # queries: query_text
               docs,                    # docs: passage_text
               labels,                  # labels: 1 or 0
               tokenizer,               # tokenizer
               max_length=512    # max_length for query and doc
               ):
      
    self.queries = queries
    self.docs = docs
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.queries)

  def __getitem__(self, idx):
    q = self.queries[idx]
    d = self.docs[idx]
    label = self.labels[idx]

    q_encoding = self.tokenizer(
        q,
        padding='max_length',
        truncation=True,
        max_length=self.max_length,
        return_tensors='pt'
    )
    
    d_encoding = self.tokenizer(
        d,
        padding='max_length',
        truncation=True,
        max_length=self.max_length,
        return_tensors='pt'
    )

    return {
        'query_input_ids': q_encoding['input_ids'].squeeze(0),
        'query_attention_mask': q_encoding['attention_mask'].squeeze(0),
        'doc_input_ids': d_encoding['input_ids'].squeeze(0),
        'doc_attention_mask': d_encoding['attention_mask'].squeeze(0),
        'label': torch.tensor(label, dtype=torch.float)
    }

class PairwiseDataset(Dataset):
  """
  构造pairwise数据集
  分别encode query和document
  """
  def __init__(self, queries, pos_docs, neg_docs, tokenizer, q_max_len=64, d_max_len=512):
    self.queries = queries
    self.pos_docs = pos_docs
    self.neg_docs = neg_docs
    self.tokenizer = tokenizer
    self.q_max_len = q_max_len
    self.d_max_len = d_max_len

  def __len__(self):
    return len(self.queries)

  def __getitem__(self, idx):
    q = self.queries[idx]
    pos = self.pos_docs[idx]
    neg = self.neg_docs[idx]

    q_enc = self.tokenizer(q, padding='max_length', truncation=True, max_length=self.q_max_len, return_tensors='pt')
    pos_enc = self.tokenizer(pos, padding='max_length', truncation=True, max_length=self.d_max_len, return_tensors='pt')
    neg_enc = self.tokenizer(neg, padding='max_length', truncation=True, max_length=self.d_max_len, return_tensors='pt')

    return {
        'q_input_ids': q_enc['input_ids'].squeeze(0),
        'q_attention_mask': q_enc['attention_mask'].squeeze(0),
        'pos_input_ids': pos_enc['input_ids'].squeeze(0),
        'pos_attention_mask': pos_enc['attention_mask'].squeeze(0),
        'neg_input_ids': neg_enc['input_ids'].squeeze(0),
        'neg_attention_mask': neg_enc['attention_mask'].squeeze(0)
    }
    
def get_triplet_train_dev_loader(data_dir,
                                 batch_size,
                                 tokenizer,
                                 max_length,
                                 subset=None):
    
    data_path = {
        'train': os.path.join(data_dir, 'train.parquet'),
        'dev': os.path.join(data_dir, 'validation.parquet'),
        'test': os.path.join(data_dir, 'test.parquet')
    }
    train_dataset = load_dataset("parquet", data_files=data_path, split='train')
    dev_dataset = load_dataset("parquet", data_files=data_path, split='dev')
    
    train_dataset = gen_triplets(train_dataset)
    dev_dataset = gen_triplets(dev_dataset)
    
    train_dataset = TripletDataset(
        queries=[i[0] for i in train_dataset],
        docs=[i[1] for i in train_dataset],
        labels=[i[2] for i in train_dataset],
        tokenizer=tokenizer,
        max_length=max_length
    )
    dev_dataset = TripletDataset(
        queries=[i[0] for i in dev_dataset],
        docs=[i[1] for i in dev_dataset],
        labels=[i[2] for i in dev_dataset],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    if subset is not None:
        subset_size = int(len(train_dataset) * subset)
        train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), size=subset_size, replace=False))
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_data_loader, dev_data_loader
    
def get_triplet_test_loader(data_dir,
                            batch_size,
                            tokenizer,
                            max_length):
    
    data_path = {
        'test': os.path.join(data_dir, 'test.parquet')
    }
    test_dataset = load_dataset("parquet", data_files=data_path, split='test')
    
    test_dataset = gen_triplets(test_dataset)
    
    test_dataset = TripletDataset(
        queries=[i[0] for i in test_dataset],
        docs=[i[1] for i in test_dataset],
        labels=[i[2] for i in test_dataset],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return test_data_loader

def get_pair_wise_train_dev_loader(data_dir,
                                   batch_size,
                                    tokenizer,
                                    q_max_len=64,
                                    d_max_len=512):
    
    data_path = {
        'train': os.path.join(data_dir, 'train.parquet'),
        'dev': os.path.join(data_dir, 'validation.parquet'),
        'test': os.path.join(data_dir, 'test.parquet')
    }
    train_dataset = load_dataset("parquet", data_files=data_path, split='train')
    dev_dataset = load_dataset("parquet", data_files=data_path, split='dev')
    
    train_dataset = gen_duples(train_dataset)
    dev_dataset = gen_duples(dev_dataset)
    
    train_dataset = PairwiseDataset(
        queries=[i[0] for i in train_dataset],
        pos_docs=[i[1] for i in train_dataset],
        neg_docs=[i[2] for i in train_dataset],
        tokenizer=tokenizer,
        q_max_len=q_max_len,
        d_max_len=d_max_len
    )
    dev_dataset = PairwiseDataset(
        queries=[i[0] for i in dev_dataset],
        pos_docs=[i[1] for i in dev_dataset],
        neg_docs=[i[2] for i in dev_dataset],
        tokenizer=tokenizer,
        q_max_len=q_max_len,
        d_max_len=d_max_len
    )
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_data_loader, dev_data_loader

def get_pair_wise_test_loader(data_dir,
                                batch_size,
                                tokenizer,
                                q_max_len=64,
                                d_max_len=512):
        
        data_path = {
            'test': os.path.join(data_dir, 'test.parquet')
        }
        test_dataset = load_dataset("parquet", data_files=data_path, split='test')
        
        test_dataset = gen_duples(test_dataset)
        
        test_dataset = PairwiseDataset(
            queries=[i[0] for i in test_dataset],
            pos_docs=[i[1] for i in test_dataset],
            neg_docs=[i[2] for i in test_dataset],
            tokenizer=tokenizer,
            q_max_len=q_max_len,
            d_max_len=d_max_len
        )
        
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return test_data_loader