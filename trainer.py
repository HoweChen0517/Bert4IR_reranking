from typing import *
import torch
import torch.nn.functional as F
from model import BertBiEncoder
import wandb
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import ndcg_score, roc_auc_score

class BiEncoderTrainer:
    """增强版训练器类，添加了NDCG评估和训练监控功能"""
    
    def __init__(
        self,
        model: BertBiEncoder,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_type: str = "contrastive",
        temperature: float = 0.1,
        use_wandb: bool = True,
        batch_size: int = 32,
        wandb_project: str = "deep-learning-project-bert4rank",
        wandb_run_name: Optional[str] = None,
        eval_ndcg_k: List[int] = [1, 3, 5]
    ):
        """
        初始化训练器
        
        Args:
            model: BiEncoder模型
            optimizer: 优化器
            device: 训练设备
            loss_type: 损失函数类型 ('contrastive', 'ranknet', 'mse', 'cross_entropy')
            temperature: 对比学习的温度参数
            use_wandb: 是否使用wandb记录训练过程
            wandb_project: wandb项目名称
            wandb_run_name: wandb运行名称，如果为None则自动生成
            eval_ndcg_k: 评估NDCG@k的k值列表
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.temperature = temperature
        self.use_wandb = use_wandb
        self.eval_ndcg_k = eval_ndcg_k
        self.batch_size = batch_size
        
        
        # 将模型移至指定设备
        self.model.to(self.device)
        
        # 初始化wandb
        if self.use_wandb:
            config = {
                "loss_type": loss_type,
                "batch_size": batch_size,
                "temperature": temperature,
                "model_type": type(model).__name__,
                "optimizer": type(optimizer).__name__,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "freeze_status": self._get_freeze_status(),
            }
            
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=config
            )
            
            # 记录模型参数状态
            param_status = {name: param.requires_grad for name, param in model.named_parameters()}
            wandb.config.update({"param_status": param_status})
    
    def _get_freeze_status(self):
        """获取模型参数冻结状态的摘要"""
        total_params = 0
        frozen_params = 0
        
        for param in self.model.parameters():
            params_in_layer = param.numel()
            total_params += params_in_layer
            if not param.requires_grad:
                frozen_params += params_in_layer
        
        return {
            "total_params": total_params,
            "frozen_params": frozen_params,
            "frozen_percentage": frozen_params / total_params * 100 if total_params > 0 else 0
        }
    
    def compute_loss(self, similarity_scores, labels):
        """
        根据选择的损失函数类型计算损失
        
        Args:
            similarity_scores: 相似度分数 [batch_size, ]
            labels: 标签 [batch_size]
            
        Returns:
            loss: 计算的损失值
        """
        if self.loss_type == "BCE":
            # 二进制交叉熵损失
            loss = F.binary_cross_entropy_with_logits(similarity_scores, labels.float())
            return loss
        elif self.loss_type == 'CE':
            # 交叉熵损失
            loss = F.cross_entropy(similarity_scores, labels.long())
            return loss
        
        else:
            raise ValueError(f"不支持的损失函数类型: {self.loss_type}")
    
    def compute_auc(self, predictions, labels):
        return {'AUC': roc_auc_score(labels, predictions)}
            
    def train_epoch(self, dataloader, epoch):
        """
        训练一个epoch，使用tqdm显示进度
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch数
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移至设备
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            doc_input_ids = batch['doc_input_ids'].to(self.device)
            doc_attention_mask = batch['doc_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            similarity_scores = self.model(
                query_input_ids, 
                query_attention_mask,
                doc_input_ids,
                doc_attention_mask
            )
            
            # 计算损失
            loss = self.compute_loss(similarity_scores, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            current_loss = loss.item()
            total_loss += current_loss
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    "train_step_loss": current_loss,
                    "train_step": epoch * len(dataloader) + batch_idx
                })
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        
        # 记录epoch级别的指标到wandb
        if self.use_wandb:
            wandb.log({
                "train_epoch_loss": avg_loss,
                "epoch": epoch
            })
        
        return avg_loss
    
    def evaluate(self, dataloader, compute_metrics=True):
        """
        评估模型，包括损失和NDCG指标
        
        Args:
            dataloader: 数据加载器
            compute_metrics: 是否计算评估指标
            
        Returns:
            results: 包含评估结果的字典
        """
        self.model.eval()
        total_loss = 0
        all_similarity_scores = []
        all_labels = []
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(dataloader, desc="Evaluation")
        
        with torch.no_grad():
            for batch in progress_bar:
                # 将数据移至设备
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                doc_input_ids = batch['doc_input_ids'].to(self.device)
                doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                similarity_scores = self.model(
                    query_input_ids, 
                    query_attention_mask,
                    doc_input_ids,
                    doc_attention_mask
                )
                
                # 计算损失
                loss = self.compute_loss(similarity_scores, labels)
                
                # 累计损失
                current_loss = loss.item()
                total_loss += current_loss
                
                # 更新进度条
                progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
                
                # 收集数据用于计算指标
                if compute_metrics:
                    all_similarity_scores.append(similarity_scores.detach().cpu())
                    all_labels.append(labels.detach().cpu())
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        
        results = {
            "loss": avg_loss
        }
        
        # 计算评估指标
        if compute_metrics and all_similarity_scores and all_labels:
            # 将所有批次的数据合并
            all_similarity_scores = torch.cat(all_similarity_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # 计算AUC指标
            auc_results = self.compute_auc(all_similarity_scores, all_labels)
            results.update(auc_results)
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log(results)
        
        return results