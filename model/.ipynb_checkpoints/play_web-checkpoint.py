from transformers import GPT2LMHeadModel

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
from matplotlib.backends.backend_webagg import FigureCanvasWebAgg
from matplotlib.backends.backend_webagg_core import WebAggApplication

import tornado

# Override the WebAgg backend's default port if needed
desired_port = 8081  # Set the desired port for WebAgg

# Load the GPT-2 model
model_hf = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
sd_hf = model_hf.state_dict()

# Watch the layers specially tockens embeddings and position embeddings
for k, v in sd_hf.items():
    print(k, v.shape)

# Print some weights
print(sd_hf['transformer.wte.weight'].view(-1)[:20])

plt.imshow(sd_hf['transformer.wte.weight'], cmap='gray')

# Get the Tornado application from the WebAgg backend
app = WebAggApplication(FigureCanvasWebAgg(fig), 'matplotlib', 'localhost', desired_port)

# Start the Tornado event loop on the specified port
print(f"WebAgg server running on http://localhost:{desired_port}")
tornado.ioloop.IOLoop.current().start()