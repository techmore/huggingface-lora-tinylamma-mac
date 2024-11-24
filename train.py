import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import json
from datasets import Dataset
import numpy as np
from datetime import datetime
import os

# Load the training data
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Format the data for instruction fine-tuning
    formatted_data = []
    for item in data:
        # Create the prompt with instruction and input
        prompt = f"### Instruction: {item['instruction']}\n\n### Input: {item['input']}\n\n### Response: {item['output']}"
        formatted_data.append({"text": prompt})
    
    return Dataset.from_dict({"text": [d["text"] for d in formatted_data]})

# Model configuration
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./tinyllama-lora-output"

# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank for adapters
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
    use_8bit_quantization=False  # Disable 8-bit quantization
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy only on non-padded tokens
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    metrics = {
        "perplexity": np.exp(np.mean(logits[mask])),
        "accuracy": np.mean(predictions == labels)
    }
    return metrics

def main():
    try:
        # Validate training data exists
        if not os.path.exists("training_data.json"):
            raise FileNotFoundError("training_data.json not found. Please generate training data first.")

        # Create state file
        try:
            with open('training_state.json', 'w') as f:
                json.dump({
                    'status': 'starting',
                    'epoch': 0,
                    'loss': 0,
                    'samples_processed': 0
                }, f)
        except IOError as e:
            print(f"Warning: Could not create training state file: {e}")
    
        print("1. Starting the training process...")
    
        # Load tokenizer
        print("2. Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            print("   Tokenizer loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
    
        # Load model
        print("3. Loading TinyLlama model...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",  # Let it automatically decide the best device
                trust_remote_code=True,
                use_cache=False,
                load_in_8bit=False,  # Disable 8-bit quantization since we're on CPU
            )
            model.config.use_cache = False  # Disable cache for training
            print("   Base model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
        # Prepare model for training
        print("4. Applying LoRA adapters...")
        try:
            model = get_peft_model(model, lora_config)
            print("   LoRA adapters applied")
        except Exception as e:
            raise RuntimeError(f"Failed to apply LoRA adapters: {e}")
    
        # Load and preprocess dataset
        print("5. Loading and preprocessing training data...")
        try:
            dataset = load_dataset("training_data.json")
            print(f"   Loaded {len(dataset)} training examples")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
    
        print("6. Tokenizing dataset...")
        try:
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            print("   Dataset tokenized")
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize dataset: {e}")
    
        # Split dataset into train and eval
        try:
            dataset = tokenized_dataset.train_test_split(test_size=0.2)
        except Exception as e:
            raise RuntimeError(f"Failed to split dataset: {e}")
    
        # Training arguments
        print("7. Setting up training arguments...")
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=4,
                num_train_epochs=10,
                learning_rate=2e-4,
                fp16=False,
                save_steps=100,
                logging_steps=10,
                max_steps=2000,
                optim="adamw_torch",
                warmup_ratio=0.1,
                group_by_length=True,
                save_safetensors=True,
                report_to="none",
                gradient_checkpointing=True,
                max_grad_norm=1.0,
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                weight_decay=0.01
            )
            print("   Training arguments set")
        except Exception as e:
            raise RuntimeError(f"Failed to set training arguments: {e}")
    
        # Initialize trainer
        print("8. Initializing trainer...")
        try:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            print("   Trainer initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize trainer: {e}")
    
        # Train
        print("9. Starting training...")
        try:
            class LogCallback(TrainerCallback):
                def on_train_begin(self, args, state, control, **kwargs):
                    with open('training_state.json', 'w') as f:
                        json.dump({
                            'status': 'training',
                            'epoch': 0,
                            'loss': 0,
                            'samples_processed': 0,
                            'start_time': datetime.now().timestamp()
                        }, f)

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        with open('training_state.json', 'w') as f:
                            json.dump({
                                'status': 'training',
                                'epoch': logs.get('epoch', 0),
                                'loss': logs.get('loss', 0),
                                'samples_processed': state.global_step * args.per_device_train_batch_size,
                                'start_time': state.start_time.timestamp() if hasattr(state, 'start_time') else datetime.now().timestamp(),
                                'learning_rate': logs.get('learning_rate', 0),
                                'global_step': state.global_step,
                                'total_steps': state.max_steps,
                            }, f)

                def on_step_end(self, args, state, control, **kwargs):
                    # Check for save_checkpoint flag
                    if os.path.exists('save_checkpoint'):
                        os.remove('save_checkpoint')
                        control.should_save = True
                        print("Emergency checkpoint requested - saving model...")

                def on_train_end(self, args, state, control, **kwargs):
                    with open('training_state.json', 'w') as f:
                        json.dump({
                            'status': 'completed',
                            'epoch': args.num_train_epochs,
                            'samples_processed': len(dataset["train"]),
                            'end_time': datetime.now().timestamp()
                        }, f)
            
            trainer.add_callback(LogCallback())
            trainer.train()
        except Exception as e:
            raise RuntimeError(f"Failed to train model: {e}")
    
        # Save final state
        try:
            with open('training_state.json', 'w') as f:
                json.dump({
                    'status': 'completed',
                    'epoch': training_args.num_train_epochs,
                    'samples_processed': len(dataset["train"])
                }, f)
        except IOError as e:
            print(f"Warning: Could not save final training state: {e}")
    
        print("   Training completed")
    
        # Save the trained model
        print("10. Saving trained model...")
        try:
            trainer.save_model()
            print("   Model saved successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
