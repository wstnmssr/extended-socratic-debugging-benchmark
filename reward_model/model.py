from trl import RewardConfig, RewardTrainer
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("wmusser/opt-1.3b-socratic-sc")
model = AutoModelForSequenceClassification.from_pretrained(
    "wmusser/opt-1.3b-socratic-sc", num_labels=1
)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_from_disk("/Users/wstnmssr/school/socratic-debugging-benchmark/reward_model/training/implicit_prompt_data")
print(dataset.num_rows)
training_args = RewardConfig(
    output_dir="opt-reward", 
    per_device_train_batch_size=2,
    
)

trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
)
print('training')
trainer.train()
