import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from typing import *

class BertBiEncoder(nn.Module):
    def __init__(
        self,
        query_encoder_name: str = "bert-base-uncased",
        doc_encoder_name: str = "bert-base-uncased", 
        share_weights: bool = False,
        freeze_query_encoder: bool = False,
        freeze_doc_encoder: bool = False,
        freeze_layers: Optional[List[int]] = None,
        pooling_strategy: str = "mean"
    ):
        """
        初始化BERT BiEncoder模型
        
        Args:
            query_encoder_name: 查询编码器使用的预训练模型名称
            doc_encoder_name: 文档编码器使用的预训练模型名称
            share_weights: 是否共享查询和文档编码器的权重
            freeze_query_encoder: 是否冻结查询编码器的所有参数
            freeze_doc_encoder: 是否冻结文档编码器的所有参数
            freeze_layers: 要冻结的特定层索引列表（如果有）
            pooling_strategy: 池化策略，可选"mean", "cls", "max"
        """
        super(BertBiEncoder, self).__init__()
        
        # 初始化查询编码器
        self.query_encoder = AutoModel.from_pretrained(query_encoder_name)
        
        # 如果共享权重，文档编码器使用与查询编码器相同的权重
        if share_weights:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = AutoModel.from_pretrained(doc_encoder_name)
        
        self.pooling_strategy = pooling_strategy
        
        # 冻结查询编码器参数（如果需要）
        if freeze_query_encoder:
            for param in self.query_encoder.parameters():
                param.requires_grad = False
        
        # 冻结文档编码器参数（如果需要）
        if freeze_doc_encoder:
            for param in self.doc_encoder.parameters():
                param.requires_grad = False
        
        # 冻结特定层（如果指定）
        if freeze_layers is not None:
            self._freeze_specific_layers(self.query_encoder, freeze_layers)
            if not share_weights:
                self._freeze_specific_layers(self.doc_encoder, freeze_layers)
    
    def _freeze_specific_layers(self, model, layer_indices):
        """冻结BERT模型中的特定层"""
        # 冻结嵌入层（如果0在层索引中）
        if 0 in layer_indices:
            for param in model.embeddings.parameters():
                param.requires_grad = False
        
        # 冻结指定的编码器层
        for idx in layer_indices:
            if idx > 0 and idx <= len(model.encoder.layer):
                for param in model.encoder.layer[idx-1].parameters():
                    param.requires_grad = False
    
    def pooling(self, model_output, attention_mask):
        """
        根据选择的策略对BERT输出进行池化
        
        Args:
            model_output: BERT模型的输出
            attention_mask: 注意力掩码
            
        Returns:
            pooled_output: 池化后的嵌入向量
        """
        if self.pooling_strategy == "cls":
            # 使用[CLS]令牌的表示
            return model_output.last_hidden_state[:, 0]
        
        elif self.pooling_strategy == "mean":
            # 对所有token的表示进行平均
            last_hidden_state = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == "max":
            # 对所有token的表示进行最大池化
            last_hidden_state = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9  # 掩码设置为非常小的值
            return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"不支持的池化策略: {self.pooling_strategy}")
    
    def encode_query(self, input_ids, attention_mask):
        """编码查询文本"""
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)    # (batch_size, seq_len, hidden_size)
        return self.pooling(outputs, attention_mask)    # (batch_size, hidden_size)
    
    def encode_doc(self, input_ids, attention_mask):
        """编码文档文本"""
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)    # (batch_size, seq_len, hidden_size)
        return self.pooling(outputs, attention_mask)    # (batch_size, hidden_size)
    
    def similarity(self, query_embeddings, doc_embeddings, metric="IP"):
        """计算内积相似度"""
        if metric == "IP":
            return torch.sum(query_embeddings * doc_embeddings, dim=1)  # (batch_size,)
        elif metric == 'cosine':
            return torch.nn.functional.cosine_similarity(query_embeddings, doc_embeddings)  # (batch_size,)
        
    
    def forward(
        self, 
        query_input_ids, 
        query_attention_mask, 
        doc_input_ids, 
        doc_attention_mask,
        return_embeddings: bool = False,
        metric: str = "IP"
    ):
        """
        前向传播函数
        
        Args:
            query_input_ids: 查询的输入ID
            query_attention_mask: 查询的注意力掩码
            doc_input_ids: 文档的输入ID
            doc_attention_mask: 文档的注意力掩码
            return_embeddings: 是否返回嵌入向量
            
        Returns:
            相似度分数和可选的嵌入向量
        """
        # 获取查询和文档的嵌入
        query_embeddings = self.encode_query(query_input_ids, query_attention_mask) # (batch_size, hidden_size)
        doc_embeddings = self.encode_doc(doc_input_ids, doc_attention_mask) # (batch_size, hidden_size)
        
        # 计算相似度分数
        similarity_scores = self.similarity(query_embeddings, doc_embeddings, metric=metric) # (batch_size,)
        
        if return_embeddings:
            return similarity_scores, query_embeddings, doc_embeddings
        else:
            return similarity_scores