from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, set_seed
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Llama 2-7B model and tokenizer
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Access model's state dictionary
state_dict = model.state_dict()

# Inspect the layers, especially token embeddings and position embeddings
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

# Print some weights (example: input embeddings)
if 'model.embed_tokens.weight' in state_dict:
    print(state_dict['model.embed_tokens.weight'].view(-1)[:20])

# Plot the positional embeddings weights
if 'model.embed_positions.weight' in state_dict:
    plt.imshow(state_dict['model.embed_positions.weight'].cpu().detach().numpy(), cmap='gray')
    plt.title("Positional Embeddings Weights")
    plt.colorbar()
    plt.show()

# Plot specific columns of positional embeddings
if 'model.embed_positions.weight' in state_dict:
    plt.plot(state_dict['model.embed_positions.weight'][:, 150].cpu().detach().numpy(), label="Column 150")
    plt.plot(state_dict['model.embed_positions.weight'][:, 200].cpu().detach().numpy(), label="Column 200")
    plt.plot(state_dict['model.embed_positions.weight'][:, 250].cpu().detach().numpy(), label="Column 250")
    plt.legend()
    plt.title("Specific Columns of Positional Embeddings")
    plt.show()

# Plot attention weights (example: first layer attention weight)
if 'model.layers.1.self_attn.q_proj.weight' in state_dict:
    plt.imshow(state_dict['model.layers.1.self_attn.q_proj.weight'][:300, :300].cpu().detach().numpy(), cmap='gray')
    plt.title("First Layer Attention Weights (300x300)")
    plt.colorbar()
    plt.show()

# Use the pipeline for text generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)  # device=0 to use the GPU
set_seed(42)
results = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

# Print generated text
for i, result in enumerate(results):
    print(f"Generated Text {i + 1}: {result['generated_text']}")