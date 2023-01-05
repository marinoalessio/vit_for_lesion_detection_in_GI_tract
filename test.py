# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:15:25 2021

@author: utente
"""
import argparse
import os
import ast
import torch
import json

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from utils.plot_conf_matrix import plot_confusion_matrix

from models.modeling import VisionTransformer, LateFusionVisionTransformer, CONFIGS
from utils.data_utils import get_loader

def setup_parser(arguments, parser):    
    for k,v in arguments.items():
        parser.add_argument(f'--{k}', type = type(v), default = v)
            
    return parser

def get_accuracy(preds, labels, accuracy_type='simple'):
    if accuracy_type == 'simple':
        return (preds == labels).mean()
    elif accuracy_type == 'balanced':
        return balanced_accuracy_score(labels, preds)
    elif accuracy_type == 'both':
        simple_accuracy = (preds == labels).mean()
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return simple_accuracy, balanced_accuracy

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.image_modality in ['T1', 'T2', 'LateFusion']:
        in_channels = 1
    elif args.image_modality in ['EarlyFusion']:
        in_channels = 2
    else :
        in_channels = 3
        
    if args.dataset in ['MRI', 'MRI-BALANCED', 'MRI-EQUAL']:
        num_classes = 4
    elif args.dataset in ['MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val', 'MRI-BALANCED-3Classes_Nested']:
        num_classes = 3
    elif args.dataset == 'cifar10':
        num_classes = 10
    else: 
        num_classes = 100
    
    if args.image_modality == 'LateFusion':
        model = LateFusionVisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, multi_stage_classification= args.multi_stage_classification, multi_layer_classification=args.multi_layer_classification, vis = True)
    else:
        model = VisionTransformer(config, args.img_size, in_channels=in_channels, zero_head=True, num_classes=num_classes, loss_weights = args.loss_weights, multi_stage_classification= args.multi_stage_classification, multi_layer_classification=args.multi_layer_classification, vis = True)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir, map_location=args.device))
    model.to(args.device)
    
    return config, args, model


def valid(args, model, test_loader, KEYS, ty):
    # Validation!
    
    model.eval()
    all_preds, all_label = [], []
    for step, batch in enumerate(test_loader):
        if args.dataset in ['MRI','MRI-BALANCED','MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val','MRI-BALANCED-3Classes_Nested', 'MRI-EQUAL']:
            x = batch[KEYS[0]].to(args.device)
            if args.image_modality == 'LateFusion':
                x2 = batch[KEYS[1]].to(args.device)
                x = (x,x2)
            y = batch[KEYS[-1]].squeeze().to(args.device).long()
        else:
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            
        with torch.no_grad():
            logits = model(x)[0]

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        
    all_preds, all_label = all_preds[0], all_label[0]
    simple_accuracy, balanced_accuracy = get_accuracy(all_preds, all_label, accuracy_type = args.accuracy)
    
    
    conf_matrix = confusion_matrix(all_label, all_preds)
    class_names = np.arange(model.num_classes)
    figure = plot_confusion_matrix(conf_matrix, class_names=class_names)
    if ty == '_simple':
        average = 'macro'
    else:
        average = 'weighted'
    prf = precision_recall_fscore_support(all_label,all_preds, average=average)
    print(prf)
    print('simple',simple_accuracy)
    print('balanced',balanced_accuracy)

    return simple_accuracy, balanced_accuracy, prf, figure
    


'''
def main():
'''   
parser = argparse.ArgumentParser()
# Required parameters
'''parser.add_argument("--test_dir", required=True,
                        help="Name of this run. Used for monitoring.")

test_dir = os.path.join(args.test_dir)'''
keys = ['fusedImage', 'label']
ty = '_simple'
test_dir = os.path.join("output","withStandarization","2021-04-28_08-35-41testVit_EarlyFusion_Nested_fold7", "inner_loop_0")
inner_loop_idx = int(test_dir.split('\\')[-1].split('_')[-1])
#test_dir = os.path.join("output","withStandarization","2021-04-28_20-47-34testViT_EarlyFusion_Nested_25patches_fold5", f"inner_loop_{inner_loop_idx }")
if test_dir[-1] == '\\':
    test_dir = test_dir[:-1]

if "Nested" in test_dir:
    file_name = test_dir.split("\\")[-2]
else:
    file_name = test_dir.split("\\")[-1]
#print(os.path.join(test_dir, file_name+'.txt'))
#Reading a .txt file
if os.path.isfile(os.path.join(test_dir, file_name+'.txt')):
    with open(os.path.join(test_dir, file_name+'.txt'), "r") as f:
        file = f.read()
    
    string_dict = file.split("model_args:")[1]
    
    string_dict = string_dict.replace('True', "'True'")
    string_dict = string_dict.replace('False', "'False'")
    string_dict = string_dict.split(", 'loss_weights")[0] + '}'
    string_dict = string_dict[2:]
    dictionary = ast.literal_eval(string_dict)
    
#Reading a .json file
elif os.path.isfile(os.path.join(test_dir, file_name+'.json')):
    with open(os.path.join(test_dir, file_name+'.json'), 'r')as fp:
        d = json.load(fp)
        dictionary = d['model_args']

else: print("error")

parser = setup_parser(dictionary, parser)

args = parser.parse_args()

args.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.output_dir = os.path.join(test_dir, "metrics")
args.pretrained_dir = os.path.join(test_dir, file_name+f'_best{ty}.bin')
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

config, args, model = setup(args)

args.eval_batch_size = 16
if not hasattr(args, 'num_patches'):
    args.num_patches = 9

_, _, loader = get_loader(args, inner_loop_idx)

simple_accuracy, balanced_accuracy, (precision, recall, f1score, support), figure = valid(args, model, loader, keys, ty)

info = {'simple_accuracy': simple_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision':precision,
        'recall': recall,
        'f1score': f1score,
        'samples':support
        }
with open(os.path.join(args.output_dir, f'metrics{ty}.json'), 'w') as fp:
    json.dump(info, fp)

plt.savefig(os.path.join(args.output_dir, f'confMat{ty}.png'))

'''
if __name__ == "__main__":
    main()
'''