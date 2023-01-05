import sys
import os
import random
import json
import math
import platform
from monai.data import CacheDataset

class FFRDataset(CacheDataset):
    def __init__(self, root_dir, split_path, section, num_fold, transforms, inner_loop = 0, seed = 100, cache_num = sys.maxsize, cache_rate=1.0, num_workers=0):    
        print(root_dir)
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.inner_loop = inner_loop
        self.text_labels = ['negative', 'positive']
        #self.transforms = transforms
        self.num_fold = num_fold
        self.seed = seed
        
        data = self._generate_data_list(split_path)
        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
           path=json.load(fp)
        data = list()
        
        if self.section == 'test':
            data = path[f'fold{self.num_fold}']['test']
        elif self.section == 'training':
            # data = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['train']
            data = path[f'fold{self.num_fold}']['train']
        elif self.section == 'validation':
            # data = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['val']
            data = path[f'fold{self.num_fold}']['val']
        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
        
        if platform.system() != 'Windows':
            for sample in data:
                for key in sample.keys():
                    if isinstance(sample[key], str):
                      sample[key] = sample[key].replace('\\', '/')
        return data     
    
    def get_label_proportions(self):
        c = [None]*8
        label_props = [None]*8
        #classes = [0, 2, 10, 22, 49, 61, 68, 69]
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(len(classes)):
            c[i] = len([el['label'] for el in self.data if el['label'] == classes[i]])
        for i in range(len(c)):
            label_props[i] = max(c)/c[i]
        
        return label_props
            
