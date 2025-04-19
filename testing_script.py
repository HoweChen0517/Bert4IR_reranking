import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AdamW
from dataset import NwayDataset, DistillModelNwayCollator
from transformers import AutoTokenizer
from model import ColBert, DenseBiEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

student_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
teacher_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

dataset = NwayDataset(
    collection_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\collection.tsv',
    queries_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\train_queries.tsv',
    train_nway_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\train_BE_1000way.tsv',
    n_way=64
)
print('数据数量：',len(dataset))
# print(dataset[0])
for one in dataset:
    if 1 in list(one[2])[-5:]:
        print(one)
        # print(type(one[0]), type(one[1]), type(one[2]))
        break
collator = DistillModelNwayCollator(student_tokenizer, teacher_tokenizer, 128, 256)
# batch = collator([dataset[i] for i in range(2)])
data_loader = DataLoader(
    dataset = dataset,
    batch_size = 2,
    collate_fn = collator,
)

# for sample in batch.get('student_doc_encodings'):
#     print(len(sample))  # n_way
#     for doc in sample:
#         print(doc['input_ids'].shape, doc['attention_mask'].shape, doc['token_type_ids'].shape)
#         print()
#         print()
#
# for sample in batch.get('teacher_inputs'):
#     print(len(sample))  # n_way
#     for pair in sample:
#         print(pair['input_ids'].shape, pair['attention_mask'].shape, pair['token_type_ids'].shape)  # [batch_size, seq_len]
#         print()  # [batch_size, seq_len]
#         print()  # [batch_size, seq_len]
        
# teacher_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
# teacher_model.eval()
# for param in teacher_model.parameters():
#     param.requires_grad = False

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# teacher_model.to(device)

# with torch.no_grad():
#     batch_logits = []
#     for sample in batch.get('teacher_inputs'):
#         logits_list = []
#         for pair in sample:
#             pair = pair.to(device)  # 将输入移动到GPU
#             # 处理教师模型的输入
#             # inputs = {
#             #     'input_ids': pair['input_ids'].to(teacher_model.device),
#             #     'attention_mask': pair['attention_mask'].to(teacher_model.device),
#             #     'token_type_ids': pair['token_type_ids'].to(teacher_model.device)
#             # }
#             # 计算教师模型的输出
#             outputs = teacher_model(**pair)
#             logits = outputs.logits
#             print(logits.shape)
#             logits_list.append(logits)
#         sample_logits = torch.cat(logits_list, dim=0).T
#         batch_logits.append(sample_logits)
#     batch_logits = torch.stack(batch_logits, dim=0)
#     batch_logits = batch_logits.squeeze(1)  # [batch_size, n_way]
#     print(batch_logits.shape)
#     print(batch_logits)
    
model = ColBert(
    model_name_or_dir='bert-base-uncased',
    teacher_model_name_or_dir='cross-encoder/ms-marco-MiniLM-L6-v2',
    n_way=64,
    use_gpu=True,
    distillation_type='soft',
    alpha=0.5,
    T=2
)

# print(model)
# 打印可以训练的参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# 打印可以训练的参数数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

print(model.device)

optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

with tqdm(total=len(data_loader), desc='Training') as pbar:
    for idx, batch in enumerate(data_loader):
        query_encodings = batch["student_query_encodings"]
        doc_encodings_list = batch["student_doc_encodings"]  # batch_size个n_way文档列表
        labels = batch["labels"]
        # print(labels)
        labels = torch.tensor(labels)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        model.to('cuda')
        # print(model(batch))
        loss, student_loss, student_scores, distill_loss, teacher_scores = model(batch)
        pbar.set_postfix(loss=loss.item())
        # print(loss, student_loss, student_scores, distill_loss, teacher_scores)
        loss.backward()
        optimizer.step()
        pbar.update(1)
        # print(batch)
        # break