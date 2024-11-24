# LLaMA 3.2 Fine-tuning with LoRA

This project implements fine-tuning of LLaMA 3.2 using Low-Rank Adaptation (LoRA) technique with the Hugging Face transformers library.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your training data in `training_data.json` in the root directory.

3. Make sure you have access to the LLaMA model on Hugging Face:
   - You need to request access to the model at https://huggingface.co/meta-llama
   - Accept the terms and conditions
   - Login to Hugging Face using `huggingface-cli login`

## Running the Training

To start the training process:
```bash
python train.py
```

The script will:
- Load and quantize the model
- Apply LoRA adapters
- Train on your dataset
- Save the resulting model in the `lora-llama-output` directory

## Configuration

You can modify the following parameters in `train.py`:
- Model configuration (model size, quantization settings)
- LoRA configuration (rank, alpha, target modules)
- Training parameters (batch size, learning rate, number of epochs)

## Output

The trained model will be saved in the `lora-llama-output` directory. You can load this model later for inference or continue training.
