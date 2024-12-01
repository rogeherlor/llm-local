# Check if CUDA is available and set the device
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the LLaMA 2 model and tokenizer once and move model to the correct device
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Flask-related imports
from flask import Flask, render_template, request

# Initialize the Flask app
app = Flask(__name__)

# Route for the homepage (input form)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        print("Received user input:", user_input)  # Debug line
        if not user_input:
            return render_template("index.html", input_text="", output_text="Please enter some text.")
        
        # Process the input
        inputs = tokenizer(user_input, return_tensors="pt").to(device)  # Move inputs to the correct device
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated output:", output_text)
        return render_template("index.html", input_text=user_input, output_text=output_text)
    
    # Handle GET request
    return render_template("index.html", input_text="", output_text="")

if __name__ == "__main__":
    # Run the app on the desired port (e.g., port 8080)
    print("Starting the Flask app... You can access it at http://localhost:8080")
    app.run(debug=False, host="172.28.0.10", port=8080)