#pip install transformers peft accelerate datasets bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import os

token = "hf_..."  # hugging faceden model için izin aldındıktan sonra yeni oluşturduğumuz tokenı buraya yapıştırıyoruz.

model_id = "meta-llama/Llama-2-7b-chat-hf" #finetuning edeceğimiz model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token) #token kısmı

# Padding hatasını önlemek için
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token=token
)

#  LoRA ile az veri ve az gpu kapasitesi ile maksimum performans görmeyi planladık.
model = get_peft_model(model, lora_config)
model.to("cuda")


# lora katmanları
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],#que ve value kullanıldı eğer eğitim verimsiz geçseydi diğer katmanlarda eklenebilirdi.
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# dataset kısmı. datasetimiz txt formatında.
with open("/content/combined_finetune_data.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n\n")

data = [{"text": line.strip()} for line in lines if line.strip()]
dataset = Dataset.from_list(data)

# Tokenizasyon işlemi
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

#  Data collator
def data_collator(features):
    input_ids = torch.tensor([f["input_ids"] for f in features])
    attention_mask = torch.tensor([f["attention_mask"] for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids
    }

# Eğitim parametreleri
training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

#  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Checkpoint kontrolü ve eğitim başlatma.(önceki model eğitimlerimizdeki hatalarımızdan ders çıkarma bölümü)
if os.path.isdir(training_args.output_dir):
    checkpoints = [
        f for f in os.listdir(training_args.output_dir)
        if f.startswith("checkpoint-") and os.path.isdir(os.path.join(training_args.output_dir, f))
    ]
    if checkpoints:
        latest = os.path.join(
            training_args.output_dir,
            sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"⏯️ Continuing from checkpoint: {latest}")
        trainer.train(resume_from_checkpoint=latest)
    else:
        print("🚀 Starting training from scratch...")
        trainer.train()
else:
    print("🚀 Starting training from scratch...")
    trainer.train()

# Eğitim sonrası adapter ağırlıklarını kaydet(colab ortamında yazıldığı için bu şekilde klasör yolu.)
adapter_dir = "./llama2-lora-finetuned/adapter"
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

print("✅ Eğitim ve model kaydı tamamlandı.")
