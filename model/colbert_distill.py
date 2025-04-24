from torch import nn
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
import torch.nn.functional as F
import wandb

class MaxSim(nn.Module):
    """
    ColBERT 的 MaxSim 机制，用来计算查询和文档之间的相似性
    """
    def __init__(self):
        super(MaxSim, self).__init__()
    
    def forward(self, query_embeddings, doc_embeddings):
        """
        计算每个query token的embedding与document中所有token的embedding的相似度
        
        Args:
            query_embeddings: [n_way, query_len, embedding_dim]
            document_embeddings: [n_way, doc_len, embedding_dim]
            
        Returns:
            colbert_score: [n_way] - 每个查询-文档对的相似度分数
        """
        # 对向量进行归一化
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)  # [n_way, query_len, embedding_dim]
        document_norm = F.normalize(doc_embeddings, p=2, dim=-1)  # [n_way, doc_len, embedding_dim]
        
        # 计算相似度矩阵
        similarity_matrix = torch.bmm(query_norm, document_norm.transpose(1, 2))  # [n_way, query_len, doc_len]
        
        # 对每个查询token取与文档最大的相似度
        max_similarity = torch.max(similarity_matrix, dim=-1)[0]  # [n_way, query_len]
        
        # 求和得到最终分数
        colbert_score = torch.sum(max_similarity, dim=-1)  # [n_way]
        
        return colbert_score

class ColBert(nn.Module):
    def __init__(self,
                 model_name_or_dir: str,
                 teacher_model_name_or_dir: str = 'cross-encoder/ms-marco-MiniLM-L6-v2',
                 n_way=64,
                 use_gpu=True,
                 distillation_type='soft',
                 alpha=0.5,
                 T=2):
        super().__init__()
        # 学生模型
        self.model = AutoModel.from_pretrained(model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        
        # 教师模型
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name_or_dir)
        # 冻结教师模型参数
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # self.student_loss_fct = nn.BCEWithLogitsLoss()
        self.student_loss_fct = nn.CrossEntropyLoss()
        self.distill_loss_fct = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.n_way = n_way
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.maxsim = MaxSim()
        self.distillation_type = distillation_type
        self.alpha = alpha  # 损失权重
        self.T = T          # 温度参数
        self.to(self.device)  # 将模型移动到GPU

        # # 初始化线性层参数
        for param in self.linear.parameters():
            if param.dim() > 1:  # 权重
                nn.init.xavier_uniform_(param)
            else:  # 偏置
                nn.init.zeros_(param)
    
    def get_teacher_scores(self, teacher_inputs):
        """
        计算教师模型的分数
        Args:
            teacher_inputs: [batch_size, seq_len]
        Returns:
            scores: [batch_size, n_way]
        """
        teacher_scores = []
        with torch.no_grad():
            for sample in teacher_inputs:
                logits_list = []
                for pair in sample:
                    # 处理教师模型的输入
                    inputs = pair.to(self.device)  # 将输入移动到GPU
                    # 计算教师模型的输出
                    outputs = self.teacher_model(**inputs)
                    logits = outputs.logits
                    # print(logits.shape)
                    logits_list.append(logits)
                sample_logits = torch.cat(logits_list, dim=0).T
                teacher_scores.append(sample_logits)
            teacher_scores = torch.stack(teacher_scores, dim=0)
            teacher_scores = teacher_scores.squeeze(1)  # [batch_size, n_way]

        return teacher_scores
                
        
    def query_encoder(self, input_ids, attention_mask):
        """查询编码
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            query_embeddings: [batch_size, seq_len, hidden_size]
        """
        # BERT编码
        outputs = self.model(input_ids, attention_mask=attention_mask)
        query_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 线性变换
        # query_embeddings = self.linear(query_embeddings)
        query_embeddings = self.linear(query_embeddings) + query_embeddings # 残差连接

        # 掩码处理
        extended_attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        query_embeddings = query_embeddings * extended_attention_mask
        
        # L2归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        
        return query_embeddings
        
    def doc_encoder(self, input_ids, attention_mask, keep_dims=True):
        """文档编码
        Args:
            input_ids: [batch_size, seq_len] 
            attention_mask: [batch_size, seq_len]
            keep_dims: bool, 是否保持维度
        Returns:
            doc_embeddings: [batch_size, seq_len, hidden_size] or list
        """
        # BERT编码
        outputs = self.model(input_ids, attention_mask=attention_mask)
        doc_embeddings = outputs.last_hidden_state
        
        # 线性变换
        doc_embeddings = self.linear(doc_embeddings)
        
        # 掩码处理
        extended_attention_mask = attention_mask.unsqueeze(-1)
        doc_embeddings = doc_embeddings * extended_attention_mask
        
        # L2归一化
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        
        if not keep_dims:
            # 移除padding tokens
            masks = attention_mask.bool()
            doc_embeddings = [doc[mask] for doc, mask in zip(doc_embeddings, masks)]
            
        return doc_embeddings

    def forward(self, student_query_encodings, student_doc_encodings, teacher_inputs, labels):
        """前向传播计算
        Args:
            batch: 数据批次，包含:
                - student_query_encodings: 查询的tokenization结果
                - student_doc_encodings: n-way文档的tokenization结果
                - labels: [batch_size, n_way] 标签
        """
        # 1. 提取输入
        labels = torch.tensor(labels)
        
        batch_size = len(student_doc_encodings)
        device = next(self.parameters()).device
        
        # 2. 查询编码
        query_vectors = self.query_encoder(
            student_query_encodings["input_ids"].to(device),
            student_query_encodings["attention_mask"].to(device)
        )  # [batch_size, query_len, hidden]
        
        # 3. 文档编码和相似度计算
        student_scores = []
        for batch_idx in range(batch_size):
            query_vector = query_vectors[batch_idx:batch_idx+1]  # [1, query_len, hidden]
            batch_scores = []
            
            # 处理当前查询的n-way文档
            for doc_encoding in student_doc_encodings[batch_idx]:
                doc_vector = self.doc_encoder(
                    doc_encoding["input_ids"].to(device),
                    doc_encoding["attention_mask"].to(device)
                )  # [1, doc_len, hidden]
                
                # 计算相似度
                score = self.maxsim(query_vector, doc_vector)  # [1]
                batch_scores.append(score)
                
            student_scores.append(torch.cat(batch_scores))  # [1, n_way]
        
        # 4. 计算损失
        student_logits = torch.stack(student_scores)  # [batch_size, n_way]
        
        # 计算student loss - CrossEntropyLoss期望原始logits作为输入
        student_loss = self.student_loss_fct(student_logits, labels.float().to(device))
        
        if self.distillation_type == "none":
            return student_loss, F.softmax(student_logits, dim=-1)
        
        elif self.distillation_type == 'soft':
            teacher_logits = self.get_teacher_scores(teacher_inputs).to(device)
            
            # 计算蒸馏损失
            student_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)
            
            distill_loss = (
                self.distill_loss_fct(
                    student_log_probs,
                    teacher_probs
                ) * (self.T * self.T)
            )
            
            loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss
            
            # 记录训练指标
            wandb.log({
                "train/loss": loss,
                "train/student_loss": student_loss,
                "train/distill_loss": distill_loss,
                "train/student_logits_mean": student_logits.mean(),
                "train/teacher_logits_mean": teacher_logits.mean(),
            })
            
            return (
                loss, 
                student_loss, 
                F.softmax(student_logits, dim=-1),  # 返回概率分布
                distill_loss,
                F.softmax(teacher_logits, dim=-1)  # 返回概率分布
            )
        
        elif self.distillation_type == 'hard':
            teacher_logits = self.get_teacher_scores(teacher_inputs).to(device)
            
            # 计算硬蒸馏损失 - 使用教师模型的预测作为硬标签
            distill_loss = F.cross_entropy(
                student_logits,
                teacher_logits.argmax(dim=-1)
            )
            
            loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss
            
            return (
                loss,
                student_loss,
                F.softmax(student_logits, dim=-1),
                distill_loss,
                F.softmax(teacher_logits, dim=-1)
            )
        else:
            raise NotImplementedError
    
    def score_pairs(self, queries, docs):
        # 验证时使用
        """
        在验证阶段，对一组
        query - doc
        对计算匹配得分。
            Args:
                queries: 一个字典，包含查询的input_ids和attention_mask，shape: [batch_size, seq_len]
                doc_encodings: 一个字典，包含文档的input_ids和attention_mask，shape: [batch_size, seq_len]

        Returns:
                scores: [batch_size]，每个query - doc对的匹配得分（float）
        """
        device = self.device

        # 编码 query 和 doc
        query_vectors = self.query_encoder(
            queries["input_ids"].to(device),
            queries["attention_mask"].to(device)
        )  # [batch_size, query_len, hidden]

        doc_vectors = self.doc_encoder(
            docs["input_ids"].to(device),
            docs["attention_mask"].to(device)
        )  # [batch_size, doc_len, hidden]

        # 计算 MaxSim 得分
        scores = self.maxsim(query_vectors, doc_vectors)  # [batch_size]

        return scores

    def save_pretrained(self, model_dir, state_dict=None):
        """
        Save the model's checkpoint to a directory
        Parameters
        ----------
        model_dir: str or Path
            path to save the model checkpoint to
        """
        self.model.save_pretrained(model_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        """
        Load model checkpoint for a path or directory
        Parameters
        ----------
        model_name_or_dir: str
            a HuggingFace's model or path to a local checkpoint
        """
        return cls(model_name_or_dir)