import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def load_hf_dataset(dataset_dir, split='train'):
    """
    Load a dataset from Hugging Face Datasets.
    :param dataset_name_or_path: The name or path of the dataset.
    :param split: The split of the dataset to load (e.g., 'train', 'test').
    :return: The loaded dataset.
    """
    assert split in ['train', 'dev', 'test'], "split must be one of ['train', 'dev', 'test']"
    data_path = {
        'train': os.path.join(dataset_dir, 'train.parquet'),
        'dev': os.path.join(dataset_dir, 'validation.parquet'),
        'test': os.path.join(dataset_dir, 'test.parquet')
    }
    dataset = load_dataset("parquet", data_files=data_path, split=split)
     
    return dataset

def gen_pairs(dataset):
    pairs = []
    for example in tqdm(dataset, total=len(dataset), desc='Processing pairs'):
        query = example["query"]
        passages = example["passages"]["passage_text"]
        labels = example["passages"]["is_selected"]
        
        pos_docs = [p for p, l in zip(passages, labels) if l == 1]
        
        for pos_doc in pos_docs:
            pairs.append((query, pos_doc))
    
    return pairs

def gen_triplets(dataset):
    """
    生成MS MARCO数据集的<query, positive_doc, negative_doc>多元组
    Return: duples: [(query, pos_doc, neg_doc), ...]
    """
    triplets = []
    for example in tqdm(dataset, total=len(dataset), desc='Processing triplets'):
      query = example["query"]
      passages = example["passages"]["passage_text"]
      labels = example["passages"]["is_selected"]

      pos_docs = [p for p, l in zip(passages, labels) if l == 1]
      neg_docs = [p for p, l in zip(passages, labels) if l == 0]

      # 构造正负配对（可采样多个 negative）
      for pos_doc in pos_docs:
          for neg_doc in neg_docs:
              triplets.append((query, pos_doc, neg_doc))

    return triplets

def read_pairs(path: str):
    """
    Read tab-delimited pairs from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of pair tuple
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'reading pairs from {Path(path).name}'):
            qid, did = line.strip().split('\t')
            pairs.append((qid, did))
    return pairs

def read_triplets(path: str):
    """
    Read tab-delimited triplets from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of triplet tuple
    """
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'reading triplets from {Path(path).name}'):
            qid, pos_id, neg_id = line.strip().split('\t')
            triplets.append((qid, pos_id, neg_id))
    return triplets

def read_nway_data(path: str, n_way=64):
    """
    读取n-way数据集
    数据集为经Bi-Encoder打分后用FAISS召回的top-100相关文档的数据集
    每行的格式为：
    query_id \t doc_id \t rank \t score \t label
    """
    # collections = read_pairs(path=r'F:\Project\Bert4IR_reranking\data\neural_ir\collection.tsv')
    # collections = [i[0] for i in collections]
    q_d_pairs = {}
    nway_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'reading n-way data from {Path(path).name}'):
            qid, did, rank, score, label = line.strip().split('\t')

            # assert did in collections, did

            if qid not in q_d_pairs:
                q_d_pairs[qid] = {
                    'related_doc': [],
                    'labels': []
                }
            q_d_pairs[qid]['related_doc'].append(did)
            q_d_pairs[qid]['labels'].append(label)


    result = {}
    # post_process
    for qid, v in q_d_pairs.items():
        if '1' not in v['labels']:
            continue
        else:
            result[qid] = v
            if len(v['related_doc']) > n_way:
                # print(v['related_doc'])
                if '1' in v['labels'][:n_way]:
                    result[qid]['related_doc'] = v['related_doc'][:n_way]
                    result[qid]['labels'] = v['labels'][:n_way]
                    continue
                else:
                    print(f'前{n_way}个搜索结果中没有包含Query@{qid}的正样本，正在从后面的文档中寻找补充到末尾')
                    related_doc = result[qid]['related_doc'][:n_way]
                    # print(related_doc)
                    labels = result[qid]['labels'][:n_way]
                    # print(labels)
                    # 如果当前qid的前n_way个文档中没有正例，需要找到refer到list的最后，找到label为1的did
                    true_docs = [(doc, label) for doc, label in zip(result[qid]['related_doc'], result[qid]['labels']) if label == '1']
                    # 如果有正例，则将正例替换为最后一个文档
                    related_doc[-len(true_docs):] = [i[0] for i in true_docs]
                    labels[-len(true_docs):] = [i[1] for i in true_docs]
                    # print(related_doc)
                    # print(labels)
                    result[qid]['related_doc'] = related_doc
                    result[qid]['labels'] = labels
                
    
    for k, v in result.items():
        nway_data.append((k, tuple(v['related_doc']), tuple(v['labels'])))
    
    return nway_data

if __name__ == '__main__':
    path = r'F:\Project\Bert4IR_reranking\data\neural_ir\train_BE_1000way.tsv'
    data = read_nway_data(path, n_way=100)
    for qid, docs, labels in data:
        if len(docs) == 0:
            print(f'<UNK>{qid}<UNK>')