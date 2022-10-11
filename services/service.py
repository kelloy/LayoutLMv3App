from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor,LayoutLMv3FeatureExtractor, TrOCRProcessor, VisionEncoderDecoderModel
from utils import utils
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from datasets import load_dataset

dataset = load_dataset("nielsr/cord-layoutlmv3")['train']

def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForTokenClassification.from_pretrained("C:/Users/keldr/Desktop/Workspace/Repo/12Sep2022/model/layoutlmv3/checkpoint-1500").to(device)
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    label_list,id2label,label2id, num_labels = utils.convert_l2n_n2l(dataset)
    width, height = image.size
    
    encoding_inputs = processor(image,return_offsets_mapping=True, return_tensors="pt",truncation = True)
    offset_mapping = encoding_inputs.pop('offset_mapping')
    for k,v in encoding_inputs.items():
        encoding_inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = model(**encoding_inputs)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding_inputs.bbox.squeeze().tolist()
    
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [utils.unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    
    return true_boxes, true_predictions
    
def text_extraction(image):
    feature_extractor = LayoutLMv3FeatureExtractor()
    encoding = feature_extractor(image, return_tensors="pt")
    
    return encoding['words'][0]


def image_render(image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    true_boxes,true_predictions = predict(image)
        
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = utils.iob_to_label(prediction)
        draw.rectangle(box, outline=utils.label_colour(predicted_label))
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=utils.label_colour(predicted_label), font=font)
    
    words = text_extraction(image)
    print(words)
    extracted_words = utils.convert_results(words,true_predictions)
    
    return image,extracted_words