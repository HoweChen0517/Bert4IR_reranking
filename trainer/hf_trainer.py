from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
import ir_measures
from ir_measures import *
import os
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
import logging

TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HFTrainer(Trainer):
    """
    继承HuggingFace Trainer类，添加了评估函数和模型保存、加载最佳模型的函数。
    
    """
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        data_collator = self.eval_collator
        eval_sampler = self._get_eval_sampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval",
    ):
        data_loader = self.get_eval_dataloader(eval_dataset)
        rerank_run = defaultdict(dict)
        self.model.eval()
        logger.info("Running evaluation")
        for batch in tqdm(
            data_loader, desc="Evaluating the model", position=0, leave=True
        ):
            qids = batch.pop("query_ids")
            dids = batch.pop("doc_ids")
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            with torch.no_grad(), torch.cuda.amp.autocast():
                scores = self.model.score_pairs(**batch).tolist()
            for qid, did, score in zip(qids, dids, scores):
                rerank_run[qid][did] = score
        self.model.train()
        qrels = (
            eval_dataset.qrels if eval_dataset is not None else self.eval_dataset.qrels
        )
        metrics = ir_measures.calc_aggregate(
            [MAP @ 10, MAP @ 20, MAP @ 100,
             nDCG @ 10, nDCG @ 20, nDCG @ 100, 
             MRR @ 10, MRR @ 20, MRR @ 100,
             RR @ 10, RR @ 20, RR @ 100,], qrels, rerank_run
        )
        metrics = {metric_key_prefix + "_" + str(k): v for k, v in metrics.items()}
        metrics["epoch"] = self.state.epoch
        self.log(metrics)
        return metrics

    def _load_best_model(self):
        logger.info(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        best_model_dir = self.state.best_model_checkpoint # 获取检查点目录路径

        weights_path = None
        is_safetensors = False
        # 优先检查 safetensors 文件是否存在
        safetensors_path = os.path.join(best_model_dir, SAFE_WEIGHTS_NAME)
        if os.path.exists(safetensors_path):
            weights_path = safetensors_path
            is_safetensors = True
            logger.info(f"Found safetensors weights file: {weights_path}")
        else:
            # 如果没有 safetensors，再检查 bin 文件
            bin_path = os.path.join(best_model_dir, WEIGHTS_NAME)
            if os.path.exists(bin_path):
                weights_path = bin_path
                logger.info(f"Found pytorch_model.bin weights file: {weights_path}")
            else:
                 # 如果两个文件都不存在，记录错误并返回
                 logger.error(f"Could not find model weights ({SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME}) in {best_model_dir}. Cannot load best model.")
                 # 根据需要可以抛出异常或直接返回
                 # raise FileNotFoundError(f"Model weights not found in {best_model_dir}")
                 return

        logger.info(f"Attempting to load model weights from {weights_path}")
        try:
            # 根据文件类型选择加载方式
            if is_safetensors:
                # 使用 safetensors 库加载
                state_dict = load_safetensors(weights_path, device="cpu")
            else:
                # 使用 torch.load 加载 .bin 文件
                state_dict = torch.load(weights_path, map_location="cpu")
        except Exception as e:
             # 捕获加载过程中可能出现的任何异常
             logger.error(f"Error loading state_dict from {weights_path}: {e}")
             return # 或者抛出异常

        # 检查加载结果是否为字典
        if not isinstance(state_dict, dict):
            logger.error(f"Loaded state_dict from {weights_path} is not a dictionary. Type: {type(state_dict)}")
            return

    def _save(self, output_dir: str = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.data_collator.tokenizer is not None:
            self.data_collator.tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.model.state_dict(), os.path.join(output_dir, TRAINING_ARGS_NAME))
