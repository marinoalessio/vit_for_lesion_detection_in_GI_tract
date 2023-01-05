import os
import json
import numpy as np
from monai.data import partition_dataset_classes
import random

'''
parameters:
'''
json_path = os.path.join('data', 'dataset_complete.json')
num_fold = 5
destination_path = os.path.join('data', f'{num_fold}BalancedFoldValSplit_complete.json')


with open(json_path) as fp:
    dataset_ = json.load(fp)

dataset_data = [i for i in range(len(dataset_))]
dataset_label = [dataset_[i]['label'] for i in range(len(dataset_))]
folds = partition_dataset_classes(dataset_data, dataset_label, num_partitions=num_fold)


data = {'num_fold':num_fold}

dataset_json = {}

for i in range(data['num_fold']):
    
    indices = np.arange(data['num_fold']).tolist()
    data[f'fold{i}']={'test':[], 'train':[], 'val':[]}
    
    #preparing the test fold
    test_fold = indices.pop(i)
    list_test = folds[test_fold]
    data[f'fold{i}']['test'] = [dataset_[f] for f in list_test]
    
    #inner loop
    for j in range(len(indices)):
        val_fold = indices[j]
        train_fold = [x for x in indices if x != val_fold]
        print(f'train:{train_fold} val:{val_fold} test:{test_fold}')
        
        # train fold
        list_train =[]
        for f in train_fold:
            list_train.extend(folds[f])
        # val fold
        list_val = folds[val_fold]
        
        # data[f'fold{i}']={'train':[], 'val':[]}
        
        data[f'fold{i}']['train'] = [dataset_[f] for f in list_train]
        data[f'fold{i}']['val'] = [dataset_[f] for f in list_val]
       

with open(destination_path, 'w') as fp:
    json.dump(data, fp) 

    