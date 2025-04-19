from tqdm import tqdm

def get_dataset_dict(dataset):
    queries = {}
    passages_corpus = {}
    passage_idx_dict = {}
    query_passage_selected = {}
    query_positive_paris = {}
    passage_count = 0
    for record in tqdm(dataset, desc="Loading all queries, document corpus and pairs"):
        query = record['query']
        query_id = record['query_id']
        if query_id not in queries:
            queries[query_id] = query
        if query_id not in query_passage_selected:
            query_passage_selected[query_id] = {}
        passages = record['passages']
        is_selected = passages['is_selected']
        passage_text = passages['passage_text']
        # print(is_selected)
        # print(passage_text)
        assert len(is_selected) == len(passage_text), f"is_selected {len(is_selected)} and passage_text {len(passage_text)} should have same length"
        for i in range(len(is_selected)):
            label = is_selected[i]
            passage = passage_text[i]
            if passage not in passages_corpus:
                passages_corpus[passage_count] = passage
                passage_idx_dict[passage] = passage_count
                passage_count += 1
            query_passage_selected[query_id][passage_idx_dict[passage]] = label
            if label:
                query_id = record['query_id']
                if query_id not in query_positive_paris:
                    query_positive_paris[query_id] = []
                query_positive_paris[query_id].append(passage_idx_dict[passage])
            
    return queries, passages_corpus, query_passage_selected, query_positive_paris
            
if __name__ == '__main__':
    from datasets import load_dataset
    import os
    data_dir =  'data/ms_marco/v1.1'
    data_path = {
        'train': os.path.join(data_dir, 'train.parquet'),
        'dev': os.path.join(data_dir, 'validation.parquet'),
        'test': os.path.join(data_dir, 'test.parquet')
    }
    # train_dataset = load_dataset("parquet", data_files=data_path, split='train')
    # dev_dataset = load_dataset("parquet", data_files=data_path, split='dev')
    test_dataset = load_dataset("parquet", data_files=data_path, split='test')
    
    queries, passages_corpus, query_passage_selected, query_positive_paris = get_dataset_dict(test_dataset)