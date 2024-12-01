from transformers import LlamaForCausalLM
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
from matplotlib.backends.backend_webagg import FigureCanvasWebAgg
import tornado.ioloop
import tornado.web

# Set the desired port for WebAgg
desired_port = 8081  # The port you want to use for the WebAgg server

# Load the LLaMA 2-7B model
model_hf = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
sd_hf = model_hf.state_dict()

# Inspect the layers, especially token embeddings and position embeddings
for k, v in sd_hf.items():
    print(k, v.shape)

# Print some weights (example: token embeddings)
if 'model.embed_tokens.weight' in sd_hf:
    print(sd_hf['model.embed_tokens.weight'].view(-1)[:20])

# Plot the token embeddings weights
fig = plt.figure()
if 'model.embed_tokens.weight' in sd_hf:
    plt.imshow(sd_hf['model.embed_tokens.weight'].cpu().detach().numpy(), cmap='gray')

# Adjust plot limits based on embedding dimensions
plt.xlim(0, 768)  # Adjust as needed
plt.ylim(0, 1024)  # Adjust as needed
plt.gca().set_aspect('auto', adjustable='box')
plt.title("Token Embeddings")

# Create a canvas for the figure
canvas = FigureCanvasWebAgg(fig)

# Define a simple Tornado handler for the WebAgg application
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "image/png")
        canvas.print_png(self)  # Sends the plot as a PNG image

# Create the Tornado application and pass the handler
application = tornado.web.Application([
    (r"/", MainHandler),
])

# Start the Tornado server on the desired port
if __name__ == "__main__":
    print(f"WebAgg server running on http://localhost:{desired_port}")
    application.listen(desired_port)  # Ensure the server listens on the correct port
    tornado.ioloop.IOLoop.current().start()
