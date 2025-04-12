import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from config import *
from dataset import get_triplet_train_dev_loader, get_triplet_test_loader
from trainer import BiEncoderTrainer
from model import BertBiEncoder
import wandb

def train_biencoder(
    train_dataloader,
    val_dataloader,
    query_encoder_name="bert-base-uncased",
    doc_encoder_name="bert-base-uncased",
    share_weights=False,
    freeze_query_encoder=False,
    freeze_doc_encoder=False,
    freeze_layers=None,
    pooling_strategy="mean",
    loss_type="contrastive",
    learning_rate=2e-5,
    num_epochs=3,
    temperature=0.1,
    device=None,
    use_wandb=True,
    wandb_project="deep-learning-project-bert4rank",
    wandb_run_name=None,
    save_model_path=None
):
    """
    训练BiEncoder模型的完整流程，增强版本
    
    Args:
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        query_encoder_name: 查询编码器名称
        doc_encoder_name: 文档编码器名称
        share_weights: 是否共享权重
        freeze_query_encoder: 是否冻结查询编码器
        freeze_doc_encoder: 是否冻结文档编码器
        freeze_layers: 要冻结的特定层
        pooling_strategy: 池化策略
        loss_type: 损失函数类型
        learning_rate: 学习率
        num_epochs: 训练轮数
        temperature: 对比学习温度
        device: 训练设备
        use_wandb: 是否使用wandb记录训练
        wandb_project: wandb项目名称
        wandb_run_name: wandb运行名称，如果未指定则自动生成
        save_model_path: 模型保存路径，如果为None则不保存
        
    Returns:
        model: 训练好的模型
        best_metrics: 最佳模型的评估指标
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果未指定wandb运行名称，则生成一个包含冻结信息的名称
    if wandb_run_name is None:
        freeze_info = ""
        if freeze_query_encoder and freeze_doc_encoder:
            freeze_info = "fully_frozen"
        elif freeze_layers is not None:
            freeze_info = f"frozen_layers_{len(freeze_layers)}"
        else:
            freeze_info = "unfrozen"
        
        wandb_run_name = f"biencoder_{pooling_strategy}_{loss_type}_{freeze_info}"
    
    # 初始化模型
    model = BertBiEncoder(
        query_encoder_name=query_encoder_name,
        doc_encoder_name=doc_encoder_name,
        share_weights=share_weights,
        freeze_query_encoder=freeze_query_encoder,
        freeze_doc_encoder=freeze_doc_encoder,
        freeze_layers=freeze_layers,
        pooling_strategy=pooling_strategy
    )
    
    # 打印模型参数状态
    print("模型参数状态概要:")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"冻结参数数量: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")
    
    # 初始化优化器 - 只优化未冻结的参数
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        # weight_decay=5e-4
    )
    
    # 初始化训练器
    trainer = BiEncoderTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_type=loss_type,
        temperature=temperature,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    
    # 训练循环
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = None
    
    print(f"开始训练 {num_epochs} 个epochs...")
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_dataloader, epoch + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        
        # 评估模型
        eval_results = trainer.evaluate(val_dataloader)
        val_loss = eval_results["loss"]
        
        # 打印评估结果
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        
        auc = eval_results["AUC"]
        print(f"Epoch {epoch + 1}/{num_epochs}, AUC: {auc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_metrics = eval_results.copy()
            print(f"找到新的最佳模型! 验证损失: {val_loss:.4f}")
            
            # 保存模型
            if save_model_path is not None:
                # 如果目录不存在，则创建
                os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': eval_results,
                    'config': {
                        'query_encoder_name': query_encoder_name,
                        'doc_encoder_name': doc_encoder_name,
                        'share_weights': share_weights,
                        'freeze_query_encoder': freeze_query_encoder,
                        'freeze_doc_encoder': freeze_doc_encoder,
                        'freeze_layers': freeze_layers,
                        'pooling_strategy': pooling_strategy,
                        'loss_type': loss_type,
                        'temperature': temperature,
                    }
                }, save_model_path)
                print(f"模型已保存到 {save_model_path}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载最佳模型，验证损失: {best_val_loss:.4f}")
    
    # 关闭wandb会话
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    # model = BiEncoder(EMBED_MODEL)
    
    train_data_loader, dev_data_loader = get_triplet_train_dev_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        subset=0.4,
    )
    
    test_data_loader = get_triplet_test_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    
    query_encoder_name = EMBED_MODEL
    doc_encoder_name = EMBED_MODEL
    share_weights = False
    freeze_query_encoder = False
    freeze_doc_encoder = False
    freeze_layers = range(0, 11)    # 冻结前11层
    pooling_strategy = "mean"
    loss_type = "BCE"
    learning_rate = 2e-5
    num_epochs = 3
    temperature = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = True

    train_biencoder(
        train_dataloader=train_data_loader,
        val_dataloader=dev_data_loader,
        query_encoder_name=query_encoder_name,
        doc_encoder_name=doc_encoder_name,
        share_weights=share_weights,
        freeze_query_encoder=freeze_query_encoder,
        freeze_doc_encoder=freeze_doc_encoder,
        freeze_layers=freeze_layers,
        pooling_strategy=pooling_strategy,
        loss_type=loss_type,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        temperature=temperature,
        device=device,
        use_wandb=use_wandb,
        wandb_project="deep-learning-project-bert4rank",
        save_model_path=os.path.join(CKPT_DIR, "biencoder_model.pth")
    )
    
    