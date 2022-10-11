import numpy as np
from datasets import ClassLabel
import const


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

def convert_l2n_n2l(dataset):
    features = dataset.features
    label_column_name = "ner_tags"

    label_list = features[label_column_name].feature.names
    if isinstance(features[label_column_name].feature, ClassLabel):
        id2label = {k:v for k,v in enumerate(label_list)}
        label2id = {v:k for k,v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset[label_column_name])
        id2label = {k:v for k,v in enumerate(label_list)}
        label2id = {v:k for k,v in enumerate(label_list)}

    return label_list, id2label, label2id, len(label_list)

def label_colour(label):
    label2color = {'MENU.PRICE':'blue', 'MENU.NM':'green', 'other':'green','MENU.TOTAL_PRICE':'red'}
    if label in label2color:
        colour = label2color.get(label)
    else:
        colour = None
    return colour

def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label

def convert_results(words,tags):
    ents = set()
    completeword = ""
    for word, tag in zip(words, tags):
        if tag != "O":
            ent_position, ent_type = tag.split("-")
            if ent_position == "S":
                ents.add((word,ent_type))
            else:
                if ent_position == "B":
                    completeword = completeword+ " "+ word
                elif ent_position == "I":
                    completeword= completeword+ " " + word
                elif ent_position == "E":
                    completeword =completeword+" " + word
                
                ents.add((completeword,ent_type))
                completeword= ""
    return ents

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]