import argparse
from transformers import TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from dataset import (
    PairDataset,
    BiEncoderPairCollator,
    NwayDataset,
    DistillModelNwayCollator,
)
from pathlib import Path
from model import ColBert
from trainer import HFDistillTrainer
import wandb

PROJECT_NAME = "deep-learning-project-bert4rank"

parser = argparse.ArgumentParser(description="Training Neural IR models")

parser.add_argument(
    "--pretrained_student_model",
    type=str,
    default="bert-base-uncased",
    help="Pretrained checkpoint for the base model",
)
parser.add_argument(
    "--pretrained_teacher_model",
    type=str,
    default="cross-encoder/ms-marco-MiniLM-L6-v2",
    help="Pretrained checkpoint for the teacher model",
)

parser.add_argument('--model',
    type=str,
    default='colbert',
    choices=['colbert'],
    help='model type'
)

parser.add_argument(
    '--n_way',
    type=int,
    default=64,
    help='Number of documents to retrieve for each query'
)
parser.add_argument(
    '--query_max_length',
    type=int,
    default=128,
    help='Maximum length of the query'
)
parser.add_argument(
    '--doc_max_length',
    type=int,
    default=256,
    help='Maximum length of the document'
)
parser.add_argument(
    '--distillation_type',
    type=str,
    default='soft',
    choices=['soft', 'hard', 'none'],
    help='Type of distillation',
)
parser.add_argument(
    '--temperature',
    type=float,
    default=0.5,
    help='Temperature for soft distillation'
)
parser.add_argument(
    '--alpha',
    type=float,
    default=0.5,
    help='Weight for the distillation loss'
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Output directory to store models and results after training",
)

parser.add_argument(
    "--epochs", type=float, default=15, help="Number of training epochs",
)
parser.add_argument(
    "--warmup_steps", type=int, default=8, help="Number of warmup steps for learning rate scheduler"
)
parser.add_argument(
    "--max_steps", type=int, default=300, help="Number of training steps"
)
parser.add_argument(
    "--train_batch_size", type=int, default=4, help="Training batch size"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=16, help="Evaluation batch size"
)
parser.add_argument(
    "--eval_steps", type=int, default=150, help="Number of eval steps"
)
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir) / args.model / "model"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

teacher_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_teacher_model)
student_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_student_model)

train_dataset = NwayDataset(
    collection_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\collection.tsv',
    queries_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\train_queries.tsv',
    train_nway_path=r'F:\Project\Bert4IR_reranking\data\neural_ir\train_BE_1000way.tsv',
    n_way=64
)
dev_dataset = PairDataset(
    collection_path="data/neural_ir/collection.tsv",
    queries_path="data/neural_ir/dev_queries.tsv",
    query_doc_pair_path="data/neural_ir/dev_bm25.trec",
    qrels_path="data/neural_ir/dev_qrels.json",
)

Nway_collator = DistillModelNwayCollator(
    student_tokenizer, teacher_tokenizer, args.query_max_length,args.doc_max_length
)
pair_collator = BiEncoderPairCollator(
    student_tokenizer, args.query_max_length, args.doc_max_length
)
model = ColBert(
    model_name_or_dir=args.pretrained_student_model,
    teacher_model_name_or_dir=args.pretrained_teacher_model,
    n_way=args.n_way,
    use_gpu=True,
    distillation_type=args.distillation_type,
    alpha=args.alpha,
    T=args.temperature,
)

wandb.init(
    project=PROJECT_NAME,
    name=f"{args.model}_{args.pretrained_teacher_model}_Distill",
    config={
        "pretrained_teacher_model": args.pretrained_teacher_model,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
    },
)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params}")
wandb.log({"model trainable params": trainable_params})

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    evaluation_strategy="steps",
    fp16=True,
    warmup_steps=args.warmup_steps,
    run_name="distill_colbert",
    metric_for_best_model="AP@10",
    load_best_model_at_end=True,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    max_steps=args.max_steps,
    save_steps=args.eval_steps,
    eval_steps=args.eval_steps,
    save_total_limit=2,
)

trainer = HFDistillTrainer(
    model,
    train_dataset=train_dataset,
    data_collator=Nway_collator,
    args=training_args,
    eval_dataset=dev_dataset,
    eval_collator=pair_collator,
)
trainer.train()
# trainer.save_model()