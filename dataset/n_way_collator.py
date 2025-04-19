from tqdm import tqdm
import torch

class DistillModelNwayCollator:
    def __init__(self, student_tokenizer, teacher_tokenizer, query_max_length, doc_max_length):
        self.tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        """
        处理批量数据，为ColBERT和Cross-Encoder准备输入
        
        Args:
            batch: [(query, docs_tuple, labels), ...]
                  每个元素包含:
                  - query: 查询文本
                  - docs_tuple: n个候选文档的元组
                  - labels: n个文档对应的相关性标签
        """
        queries, docs_lists, labels = zip(*batch)
        batch_size = len(queries)
        n_way = len(docs_lists[0])  # 每个查询的候选文档数

        # 1. 处理ColBERT(student)的输入
        # 处理查询
        student_query_encodings = self.tokenizer(
            list(queries),
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt"
        )

        # 处理文档 - 保持独立的文档编码以便逐个计算相似度
        student_doc_encodings = []
        for docs_tuple in docs_lists:
            # 对每个查询的n_way文档分别编码
            docs_encoding = [
                self.tokenizer(
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=self.doc_max_length,
                    return_tensors="pt"
                ) for doc in docs_tuple
            ]
            student_doc_encodings.append(docs_encoding)

        # 2. 处理Cross-Encoder(teacher)的输入
        # 为每个(query, doc)对准备输入
        teacher_inputs = []
        for query, docs_tuple in zip(queries, docs_lists):  # query和对应的n_way个文档
            query_doc_pairs = []
            for doc in docs_tuple:
                # 根据cross-encoder的输入要求拼接query和doc
                encoded = self.teacher_tokenizer(
                    query,
                    doc,
                    padding=True,
                    truncation=True,
                    max_length=self.query_max_length + self.doc_max_length,
                    return_tensors="pt"
                )
                query_doc_pairs.append(encoded)
            teacher_inputs.append(query_doc_pairs)

        return {
            # ColBERT输入
            "student_query_encodings": student_query_encodings, # [batch_size, seq_len]
            "student_doc_encodings": student_doc_encodings, # [batch_size, n_way, seq_len]
            
            # Cross-Encoder输入
            "teacher_inputs": teacher_inputs,   # [batch_size, n_way, (query, doc) encoding]
            
            # 标签
            "labels": labels
        }
        