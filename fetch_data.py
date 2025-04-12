# 准备huggingface上面的MS MARCO数据集

import os
from datasets import load_dataset
from config import DATA_DIR

data_path = {
    'train': os.path.join(DATA_DIR, 'train.parquet'),
    'dev': os.path.join(DATA_DIR, 'validation.parquet'),
    'test': os.path.join(DATA_DIR, 'test.parquet')
}
train_dataset = load_dataset("parquet", data_files=data_path, split='train')
dev_dataset = load_dataset("parquet", data_files=data_path, split='dev')
test_dataset = load_dataset("parquet", data_files=data_path, split='test')

print(f"train_dataset: {len(train_dataset)}")
print(f"dev_dataset: {len(dev_dataset)}")
print(f"test_dataset: {len(test_dataset)}")