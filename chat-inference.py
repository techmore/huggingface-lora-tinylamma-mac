import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model():
    print("Loading model...")
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "cpu",
        trust_remote_code=True
    )
    
    # Load our trained LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        "tinyllama-lora-output",
        torch_dtype=torch.float32,
        device_map="mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text=""):
    # Format the prompt
    prompt = f"### Instruction: {instruction}\n\n### Input: {input_text}\n\n### Response:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = response.split("### Response:")[-1].strip()
    return response

def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    print("Model loaded! Enter your questions (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        instruction = input("\nEnter instruction: ")
        if instruction.lower() == 'quit':
            break
            
        input_text = input("Enter input (optional, press Enter to skip): ")
        
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, instruction, input_text)
        print("\nResponse:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()
