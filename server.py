from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import os
import numpy as np
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

print(torch.cuda.is_available())

from train_log.RIFE_HDv3 import Model
RIFE = Model()
RIFE.load_model("./train_log/", -1)
print("Loaded v3.x HD model.")

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
netscale = 2
file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'

ROOT_DIR = os.path.dirname(os.path.curdir)
model_path = load_file_from_url(url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

upsampler = RealESRGANer(
    scale=2,
    model_path=model_path,
    model=model,
    gpu_id=0)


app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_image():
    # Get the image file from the request
    image_file = request.files['image']
    
    # Load the image using OpenCV
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Process the image as needed
    # processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image, _ = upsampler.enhance(image)
    
    # Convert the processed image to byte stream
    _, encoded_image = cv2.imencode('.png', processed_image)
    
    # Create an in-memory file-like object
    img_io = io.BytesIO(encoded_image.tobytes())
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run()
