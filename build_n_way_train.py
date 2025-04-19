import os
import json
import argparse
from index import Faiss
from model.dense_encoder import DenseBiEncoder
from utils import write_trec_run, write_n_way_tsv
from utils.dataset_utils import read_pairs
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict

parser = argparse.ArgumentParser(description="Ranking with BiEncoder")
parser.add_argument(
    "--c", type=str, default="data/neural_ir/collection.tsv", help="path to document collection"
)
parser.add_argument(
    "--q", type=str, default="data/neural_ir/train_queries.tsv", help="path to queries"
)
parser.add_argument(
    "--ground_truth", type=str, default="data/neural_ir/train_qrels.json", help="path to ground truth"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/dense/model/checkpoint-4000",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="data/neural_ir/",
    help="path to output run file",
)
parser.add_argument(
    '--search_k',
    type=int,
    default=1000,
    help='number of documents to retrieve for each query',
)
args = parser.parse_args()

docs = read_pairs(args.c)
queries = read_pairs(args.q)

collection = {}
with open(args.c, "r", encoding='utf-8') as f:
    for line in f:
        id_, text = line.rstrip("\n").split("\t")
        collection[id_] = text
doc_ids_ = list(collection.keys())

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = DenseBiEncoder.from_pretrained(args.checkpoint).to(args.device)
model.eval()
query_embs = []
docs_embs = []
doc_ids = []
for idx in tqdm(
    range(0, len(docs), args.bs), desc="Encoding documents", position=0, leave=True
):
    batch = docs[idx : idx + args.bs]
    docs_texts = [e[1] for e in batch]

    doc_ids.extend([e[0] for e in batch])
    docs_inps = tokenizer(
        docs_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_embs = model.encode(**docs_inps).to("cpu")
        docs_embs.append(batch_embs)

index = Faiss(d=docs_embs[0].size(1))
docs_embs = torch.cat(docs_embs, dim=0).numpy().astype("float32")
index.add(docs_embs)
# ?for batch_embds in tqdm(docs_embs, desc="Indexing document embeddings"):
# index.add(batch_embs.numpy().astype("float32"))

with open(args.ground_truth, "r") as f:
    qrels = json.load(f)
qrels = {k: set(v) for k, v in qrels.items()}

run = defaultdict(list)
queries_embs = []
right_hit = 0
for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = (
            model.encode(**query_inps).to("cpu").numpy().astype("float32")
        )
    scores, docs_idx = index.search(batch_query_embs, args.search_k)    # Faiss 近似搜索->List[scores], List[ids]
    for idx in range(len(batch)):
        query_id = batch[idx][0]
        ground_truth = qrels.get(query_id, [])
        find = False
        for i, score in zip(docs_idx[idx], scores[idx]):
            if i < 0:
                continue
            doc_id = doc_ids[i]

            # 纠错
            assert doc_id in doc_ids_, doc_id

            if doc_id in ground_truth:
                run[query_id].append((doc_id, score, 1))
                find = True
                right_hit += 1
            else:
                run[query_id].append((doc_id, score, 0))
        if not find:    # search top k 没找到的处理
            for doc_id in ground_truth:
                run[query_id].append((doc_id, -1, 1))

print(f"right hit: {right_hit}")
print(f"total query: {len(qrels)}")
print(f'hit rate@{args.search_k}: {right_hit / len(qrels)}')

# output_path = os.path.join(args.o, f"train_BE_{args.search_k}way.tsv")
# write_trec_run(run, args.o)
# write_n_way_tsv(run, output_path)