import gradio as gr
import torch
from datasets import load_dataset
from services import service
from const import const
import os
import pytesseract

#os.system('pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu')
#os.system('pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR\\tesseract.exe"
#os.system('sudo apt-get install tesseract-ocr')
#os.system('pip install -q pytesseract')
print("pytesseract:",pytesseract.__version__)

examples = [['./examples/example1.png'],['./examples/example2.png'],['./examples/example3.png']]

demo = gr.Interface(fn = service.image_render,
                    inputs = gr.inputs.Image(type="pil"),
                    outputs = [gr.outputs.Image(type="pil", label="annotated image"),'text'],
                    css = const.css,
                    examples = examples,
                    allow_flagging=True,
                    flagging_options=["incorrect", "correct"],
                    flagging_callback = gr.CSVLogger(),             
                    flagging_dir = "flagged"
                   )

if __name__ == "__main__":
    demo.launch()