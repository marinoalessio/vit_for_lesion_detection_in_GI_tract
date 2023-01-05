from glob import glob
import os 
import json

folders = glob(os.path.join('data','dataset_SB3','*'))

dataset = []

for folder in folders:
	label = folder.split("\\")[-1]
	images = glob(os.path.join(folder, '*'))
	for image in images:
		d = {'image' : image, 'label': int(label)}
		dataset.append(d)

with open(os.path.join('data', 'dataset_complete.json'), 'w') as f:
	json.dump(dataset, f)