import logging

import torch
import math

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.transforms import ResizeWithRatio, Delete4Ch,Convert1Ch, NewMergedImage, RandPatchedImageLateFusion,CenterPatchedImageAndEarlyFusion, RandPatchedImage3Channels, RandPatchedImage, RandPatchedImageAndEarlyFusion, RandDepthCrop, NDITKtoNumpy
from monai.transforms import (
    AddChannelD, 
    Compose, 
    LoadImageD, 
    NormalizeIntensityD ,
    OrientationD, 
    RandFlipD, 
    RandRotateD, 
    ResizeD, 
    ScaleIntensityD, 
    ToTensorD, 
    AsChannelFirstD, 
    DivisiblePadD, 
    RandRotate90D)
import os
import platform
import numpy as np
from dataset.FFRDatasetCV import FFRDataset

logger = logging.getLogger(__name__)


def get_loss_weights(dataset, multi_stage_classification = False):
    # weights = [0.032, 0.42, 0.57, 0.13, 1., 0.88, 0.047, 0.31]
    # weights = [-1]
    if dataset in ["MRI-BALANCED", "MRI-EQUAL"]:
        weights = [0.68, 0.43, 1., 0.44]
    elif dataset in ['MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val', 'MRI-BALANCED-3Classes_Nested']:
        if multi_stage_classification:
            weights = [[0.87, 1.],[1., 0.64]]
        else:
            weights = [1., 0.65, 0.45]
    return torch.tensor(weights)
            
def get_loader(args, inner_loop_idx = None):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    KEYS = ('image', "label")
    train_transforms = Compose([
        LoadImageD(KEYS[0]),
        Delete4Ch(KEYS[0]),
        ResizeWithRatio(KEYS[0], image_size = args.img_size),
        AsChannelFirstD(KEYS[0]),
        ScaleIntensityD(KEYS[0]),
        DivisiblePadD(KEYS[0], k = args.img_size, mode = 'constant'),
        RandFlipD(KEYS[0], prob = 0.5, spatial_axis=0),
        RandRotate90D(KEYS[0], prob=0.5, spatial_axes=(0,1)),  
        ToTensorD(KEYS[0])
        ])


    val_transforms = Compose([
        LoadImageD(KEYS[0]),
        Delete4Ch(KEYS[0]),
        ResizeWithRatio(KEYS[0], image_size = args.img_size),
        AsChannelFirstD(KEYS[0]),
        ScaleIntensityD(KEYS[0]),
        DivisiblePadD(KEYS[0], k = args.img_size, mode = 'constant'),
        ToTensorD(KEYS[0]),
        ])
    
    dataset = {}
    dataset["train"] = FFRDataset(root_dir = args.root_dir, split_path = args.split_path, section = 'training', num_fold = args.num_fold, transforms = train_transforms, inner_loop = args.inner_loop)
    dataset["validation"] = FFRDataset(root_dir = args.root_dir, split_path = args.split_path, section = 'validation', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop)
    dataset["test"] = FFRDataset(root_dir = args.root_dir, split_path = args.split_path, section = 'test', num_fold = args.num_fold, transforms = val_transforms, inner_loop = args.inner_loop)
    
    loss_weights = torch.Tensor(dataset["train"].get_label_proportions())
    print('loss_weights: ', loss_weights)
    
    train_loader = DataLoader(dataset["train"],
                              batch_size = args.train_batch_size,
                              num_workers=0,
                              shuffle=True
                              )
    val_loader = DataLoader(dataset["validation"],
                              batch_size = args.eval_batch_size,
                              num_workers=0,
                              )
    test_loader = DataLoader(dataset["test"],
                              batch_size = args.eval_batch_size,
                              num_workers=0,
                              )
    return train_loader, val_loader, test_loader, loss_weights

'''
def get_loader(args, inner_loop_idx = None):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "MRI":
        from dataset.PanDatasetCV import PanDataset
        data_dir = os.path.join('data')
        split_path = os.path.join('data', '10FoldMRIcrossValSplit.json')
        
        if args.image_modality== 'T1':
            KEYS = ('image_T1', 'label')
        elif args.image_modality== 'T2':
            KEYS = ('image_T2', 'label')
        elif args.image_modality== 'EarlyFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            
        SECTIONS = ('training', 'validation', 'test')
        train_transforms = Compose([
            LoadImageD(keys=[KEYS[0]]),
            AddChannelD(keys=[KEYS[0]]),
            ScaleIntensityD(keys=[KEYS[0]]),
            OrientationD(keys=[KEYS[0]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0]], spatial_size=(-1,64,64)),
            RandRotateD(keys=KEYS[0], range_x = np.pi, prob = 1, keep_size=True),
            RandFlipD(keys = KEYS[0], prob = 0.5, spatial_axis = 0),
            ToTensorD(KEYS),
            RandPatchedImage(KEYS, num_patches = 9)
            ])
        
        test_transforms = Compose([
            LoadImageD(keys=[KEYS[0]]),
            AddChannelD(keys=[KEYS[0]]),
            ScaleIntensityD(keys=[KEYS[0]]),
            OrientationD(keys=[KEYS[0]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0]], spatial_size=(-1,64,64)),
            ToTensorD(KEYS),
            RandPatchedImage(KEYS, num_patches = 9)
            ])
        
        trainset = PanDataset(root_dir=data_dir, 
                              split_path=split_path, 
                              section=SECTIONS[0], 
                              num_fold=args.num_fold, 
                              transforms=train_transforms)
        valset = PanDataset(root_dir=data_dir, 
                              split_path=split_path, 
                              section=SECTIONS[1], 
                              num_fold=args.num_fold,
                              transforms=test_transforms)
        testset = PanDataset(root_dir=data_dir, 
                              split_path=split_path, 
                              section=SECTIONS[2], 
                              num_fold=args.num_fold,
                              transforms=test_transforms)
    
    elif args.dataset == "MRI-EQUAL":
        from monai.data import Dataset
        import json
        data_dir = os.path.join('data')
        split_path = os.path.join('data', 'FoldMRIcrossValSplitBALANCED.json')
        
        
        if args.image_modality == 'T1':
            KEYS = ('image_T1', 'label')
            patched_transf = RandPatchedImage(KEYS, num_patches = 9)
        elif args.image_modality == 'T2':
            KEYS = ('image_T2', 'label')
            patched_transf = RandPatchedImage(KEYS, num_patches = 9)
        elif args.image_modality == 'EarlyFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf = RandPatchedImageAndEarlyFusion(KEYS, num_patches = 9)
        
        train_transforms = Compose([
            LoadImageD(keys=[KEYS[0], KEYS[1]]),
            AddChannelD(keys=[KEYS[0], KEYS[1]]),
            ScaleIntensityD(keys=[KEYS[0], KEYS[1]]),
            OrientationD(keys=[KEYS[0], KEYS[1]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0], KEYS[1]], spatial_size=(-1,64,64)),
            RandRotateD(keys=[KEYS[0], KEYS[1]], range_x = np.pi, prob = 1, keep_size=True),
            RandFlipD(keys = [KEYS[0], KEYS[1]], prob = 0.5, spatial_axis = 0),
            ToTensorD(KEYS),
            patched_transf
            ])
        
        test_transforms = Compose([
            LoadImageD(keys=[KEYS[0], KEYS[1]]),
            AddChannelD(keys=[KEYS[0], KEYS[1]]),
            ScaleIntensityD(keys=[KEYS[0], KEYS[1]]),
            OrientationD(keys=[KEYS[0], KEYS[1]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0], KEYS[1]], spatial_size=(-1,64,64)),
            ToTensorD(KEYS),
            patched_transf
            ])
        
        with open(split_path) as fp:
            dataset_ = json.load(fp)
        
        trainset = Dataset(data=dataset_['train'], transform=train_transforms)
        valset = Dataset(data=dataset_['val'], transform=test_transforms)
        testset = Dataset(data=dataset_['test'], transform=test_transforms)
        
        train_loader = DataLoader(trainset,
                                  batch_size = args.train_batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        val_loader = DataLoader(valset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        test_loader = DataLoader(testset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        return train_loader, val_loader, test_loader
        
    elif args.dataset == "MRI-BALANCED":
        from monai.data import Dataset
        import json
        data_dir = os.path.join('data')
        split_path = os.path.join('data', '10BalancedFoldMRIcrossValSplit.json')
        
        if args.image_modality== 'T1':
            KEYS = ('image_T1', 'label')
        elif args.image_modality== 'T2':
            KEYS = ('image_T2', 'label')
        elif args.image_modality== 'EarlyFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
        
        train_transforms = Compose([
            LoadImageD(keys=[KEYS[0]]),
            AddChannelD(keys=[KEYS[0]]),
            ScaleIntensityD(keys=[KEYS[0]]),
            OrientationD(keys=[KEYS[0]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0]], spatial_size=(-1,64,64)),
            RandRotateD(keys=KEYS[0], range_x = np.pi, prob = 1, keep_size=True),
            RandFlipD(keys = KEYS[0], prob = 0.5, spatial_axis = 0),
            ToTensorD(KEYS),
            RandPatchedImage(KEYS, num_patches = 9)
            ])
        
        test_transforms = Compose([
            LoadImageD(keys=[KEYS[0]]),
            AddChannelD(keys=[KEYS[0]]),
            ScaleIntensityD(keys=[KEYS[0]]),
            OrientationD(keys=[KEYS[0]], axcodes="RAS"),
            ResizeD(keys=[KEYS[0]], spatial_size=(-1,64,64)),
            ToTensorD(KEYS),
            RandPatchedImage(KEYS, num_patches = 9)
            ])
        
        with open(split_path) as fp:
            dataset_ = json.load(fp)
        
        
        trainset = Dataset(data=dataset_[f'fold{args.num_fold}']['train'], transform=train_transforms)
        valset = Dataset(data=dataset_[f'fold{args.num_fold}']['val'], transform=test_transforms)
        testset = Dataset(data=dataset_[f'fold{args.num_fold}']['test'], transform=test_transforms)
        
        train_loader = DataLoader(trainset,
                                  batch_size = args.train_batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        val_loader = DataLoader(valset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        test_loader = DataLoader(testset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        return train_loader, val_loader, test_loader
    elif args.dataset in ['MRI-BALANCED-3Classes', 'MRI-BALANCED-3Classes_2Val']:
        from monai.data import CacheDataset, DataLoader
        import json
        data_dir = os.path.join('data')
        if args.dataset == 'MRI-BALANCED-3Classes':
            split_path = os.path.join('data', '10BalancedFoldMRIcrossValSplit_3Class.json')
        else:
            split_path = os.path.join('data', '10BalancedFoldMRIcrossValSplit_3Class_2Val.json')
        
        patch_per_side = int(math.sqrt(args.num_patches))
        spatial_size = (-1, int(args.img_size/patch_per_side), int(args.img_size/patch_per_side))
        
        if args.image_modality == 'T1':
            KEYS = ('image_T1', 'label')
            patched_transf_train = [RandPatchedImage(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T2':
            KEYS = ('image_T2', 'label')
            patched_transf_train = [RandPatchedImage(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'EarlyFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [RandPatchedImageAndEarlyFusion(KEYS, num_patches = args.num_patches)]
            patched_transf_test = [CenterPatchedImageAndEarlyFusion(KEYS, num_patches = args.num_patches)]
        elif args.image_modality == 'LateFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [RandPatchedImageLateFusion(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'EarlyFusion3Channels':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [NewMergedImage(keys=KEYS[:-1]), RandPatchedImage3Channels(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T1_nopatched':
            KEYS = ('image_T1', 'label')
            spatial_size = (-1,args.img_size,args.img_size)
            patched_transf_train = [RandDepthCrop(KEYS, num_slices = 3)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T2_nopatched':
            KEYS = ('image_T2', 'label')
            spatial_size = (-1,args.img_size,args.img_size)
            patched_transf_train = [RandDepthCrop(KEYS, num_slices = 3)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        
        train_list = [
            LoadImageD(keys = KEYS[:-1]),
            AddChannelD(keys = KEYS[:-1]),
            NDITKtoNumpy(keys=KEYS[:-1]),
            NormalizeIntensityD(keys = KEYS[:-1]),
            ScaleIntensityD(keys = KEYS[:-1]),
            OrientationD(keys = KEYS[:-1], axcodes="RAS"),
            ResizeD(keys = KEYS[:-1], spatial_size=spatial_size),
            RandRotateD(keys = KEYS[:-1], range_x = np.pi, prob = 1, keep_size=True),
            RandFlipD(keys = KEYS[:-1], prob = 0.5, spatial_axis = 0),
            ToTensorD(KEYS)
            ] + patched_transf_train
        train_transforms = Compose(train_list)
        
        test_list =[
            LoadImageD(keys = KEYS[:-1]),
            AddChannelD(keys = KEYS[:-1]),
            NDITKtoNumpy(keys=KEYS[:-1]),
            NormalizeIntensityD(keys = KEYS[:-1]),
            ScaleIntensityD(keys = KEYS[:-1]),
            OrientationD(keys = KEYS[:-1], axcodes="RAS"),
            ResizeD(keys = KEYS[:-1], spatial_size=spatial_size),
            ToTensorD(KEYS)
            ] + patched_transf_test
        test_transforms = Compose(test_list)
        
        with open(split_path) as fp:
            dataset_ = json.load(fp)
        
        
        dataset={
            'train' : dataset_[f'fold{args.num_fold}']['train'],
            'val' : dataset_[f'fold{args.num_fold}']['val'],
            'test' : dataset_[f'fold{args.num_fold}']['test'],
            }        
        
        
        if platform.system() != 'Windows':
            for split in dataset:
                for sample in dataset[split]:
                    for key in sample.keys():
                        if type(sample[key]) == str:
                            sample[key] = sample[key].replace('\\', '/')
                            
        trainset = CacheDataset(dataset['train'], transform=train_transforms)
        valset = CacheDataset(dataset['val'], transform=test_transforms)
        testset = CacheDataset(dataset['test'], transform= test_transforms)
        
        
        train_loader = DataLoader(trainset,
                                  batch_size = args.train_batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        val_loader = DataLoader(valset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        test_loader = DataLoader(testset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        return train_loader, val_loader, test_loader    
    elif args.dataset in ['MRI-BALANCED-3Classes_Nested']:
        from monai.data import CacheDataset, DataLoader
        import json
        data_dir = os.path.join('data')
        split_path = os.path.join('data', '10BalancedFoldMRINestedCrossValSplit_3Class.json')
        
        patch_per_side = int(math.sqrt(args.num_patches))
        spatial_size = (-1, int(args.img_size/patch_per_side), int(args.img_size/patch_per_side))
        
        if args.image_modality == 'T1':
            KEYS = ('image_T1', 'label')
            patched_transf_train = [RandPatchedImage(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T2':
            KEYS = ('image_T2', 'label')
            patched_transf_train = [RandPatchedImage(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'EarlyFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [RandPatchedImageAndEarlyFusion(KEYS, num_patches = args.num_patches)]
            patched_transf_test = [CenterPatchedImageAndEarlyFusion(KEYS, num_patches = args.num_patches)]
        elif args.image_modality == 'LateFusion':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [RandPatchedImageLateFusion(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'EarlyFusion3Channels':
            KEYS = ('image_T1', 'image_T2', 'label')
            patched_transf_train = [NewMergedImage(keys=KEYS[:-1]), RandPatchedImage3Channels(KEYS, num_patches = args.num_patches)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T1_nopatched':
            KEYS = ('image_T1', 'label')
            spatial_size = (-1,args.img_size,args.img_size)
            patched_transf_train = [RandDepthCrop(KEYS, num_slices = 3)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        elif args.image_modality == 'T2_nopatched':
            KEYS = ('image_T2', 'label')
            spatial_size = (-1,args.img_size,args.img_size)
            patched_transf_train = [RandDepthCrop(KEYS, num_slices = 3)]
            patched_transf_test = patched_transf_train #MUST FIX IT
        
        train_list = [
            LoadImageD(keys = KEYS[:-1]),
            AddChannelD(keys = KEYS[:-1]),
            NDITKtoNumpy(keys=KEYS[:-1]),
            NormalizeIntensityD(keys = KEYS[:-1]),
            ScaleIntensityD(keys = KEYS[:-1]),
            OrientationD(keys = KEYS[:-1], axcodes="RAS"),
            ResizeD(keys = KEYS[:-1], spatial_size=spatial_size),
            RandRotateD(keys = KEYS[:-1], range_x = np.pi, prob = 1, keep_size=True),
            RandFlipD(keys = KEYS[:-1], prob = 0.5, spatial_axis = 0),
            ToTensorD(KEYS)
            ] + patched_transf_train
        train_transforms = Compose(train_list)
        
        test_list =[
            LoadImageD(keys = KEYS[:-1]),
            AddChannelD(keys = KEYS[:-1]),
            NDITKtoNumpy(keys=KEYS[:-1]),
            NormalizeIntensityD(keys = KEYS[:-1]),
            ScaleIntensityD(keys = KEYS[:-1]),
            OrientationD(keys = KEYS[:-1], axcodes="RAS"),
            ResizeD(keys = KEYS[:-1], spatial_size=spatial_size),
            ToTensorD(KEYS)
            ] + patched_transf_test
        test_transforms = Compose(test_list)
        
        with open(split_path) as fp:
            dataset_ = json.load(fp)
                
        dataset={
            'train' :dataset_[f'fold{args.num_fold}'][f'inner{inner_loop_idx}']['train'],
            'val' : dataset_[f'fold{args.num_fold}'][f'inner{inner_loop_idx}']['val'],
            'test' : dataset_[f'fold{args.num_fold}']['test'],
            }        
        
        
        if platform.system() != 'Windows':
            for split in dataset:
                for sample in dataset[split]:
                    for key in sample.keys():
                        if type(sample[key]) == str:
                            sample[key] = sample[key].replace('\\', '/')
                            
        trainset = CacheDataset(dataset['train'], transform=train_transforms)
        valset = CacheDataset(dataset['val'], transform=test_transforms)
        testset = CacheDataset(dataset['test'], transform= test_transforms)
        
        
        train_loader = DataLoader(trainset,
                                  batch_size = args.train_batch_size,
                                  num_workers=0,
                                  shuffle=True
                                  )
        val_loader = DataLoader(valset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        test_loader = DataLoader(testset,
                                  batch_size = args.eval_batch_size,
                                  num_workers=0,
                                  )
        return train_loader, val_loader, test_loader
        
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return train_loader, val_loader ,test_loader
'''