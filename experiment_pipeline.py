import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import json
import torch
from tqdm import tqdm

from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

from eval import evaluate, bootstrap, plot_roc
from zero_shot import make as val_make, CXRTestDataset, make_true_labels, run_softmax_eval
from run_train_multi import make as train_make
from run_train import train_batch, train_log, save
from train import preprocess_text


class Config(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])
            
config = {
    'train_cxr_folder': 'data/mimic-cxr-data/h5files/train-smoke-test/',
    'train_txt_folder': 'data/mimic-cxr-data/reports/train-smoke-test/',
    'val_cxr_filepath': 'data/chexpert-test/chexpert-test/chexlocalize/CheXpert/chexpert_test.h5',
    'val_groundtruth': 'data/chexpert-test/groundtruth.csv',
    'batch_size': 64,
    'epochs': 10,
    'lr': 5e-5,
    'log_interval': 100,
    'save_dir': 'data/chexzero-experiments/',
    'seed': 1234,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'context_length': 77,
    'random_init': False,
    'model_name': 'experiment_SMOKE_TEST'
}

config = Config(config)


cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

cxr_pair_template: Tuple[str] = ("{}", "no {}")


def experiment_pipeline(
    cxr_labels,
    cxr_pair_template,
    config
): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    # save config dictionary
    config_path = os.path.join(model_save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f)
    
    # create train objects
    model, train_loaders, device, criterion, optimizer = train_make(config)
    
    # create validation objects
    transformations = [
        # means computed from sample in `cxr_stats` notebook
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
    ]
    input_resolution = 224
    transformations.append(Resize(input_resolution, interpolation=InterpolationMode.BICUBIC))
    transform = Compose(transformations)
    val_dset = CXRTestDataset(img_path=config.val_cxr_filepath, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dset, shuffle=False)
    
    # Run training
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    best_roc_auc = -np.inf
    
    train_total_batches = [len(train_loader) for train_loader in train_loaders]
    total_batches = config.epochs * len(train_loaders) * np.sum(train_total_batches)
    
    batch_train_losses = []
    batch_val_roc_auc = []
    y_true = make_true_labels(cxr_true_labels_path=config.val_groundtruth, cxr_labels=cxr_labels)
    
    for e, epoch in enumerate(range(config.epochs)):
        running_loss = 0.0 # running loss over batch
        
        for t, train_loader in enumerate(train_loaders):
            for b, data in enumerate(train_loader):
                
                print(f'training epoch {e+1}/{config.epochs} | train loader {t+1}/{len(train_loaders)} | batch {b+1}/{len(train_loader)} | total batch {batch_ct+1}/{total_batches}')
                      
                # get the images
                images = data['img']

                texts = data['txt']
                texts = preprocess_text(texts, model) 

                # perform step for a single batch
                loss = train_batch(images, texts, model, device, criterion, optimizer)
                example_ct +=  len(images)
                batch_ct += 1
                running_loss += loss.item()
                
                # save current batch's train loss
                batch_train_losses.append(loss.item())
                batch_train_losses_path = os.path.join(model_save_dir, "batch_train_losses.json")
                with open(batch_train_losses_path, 'w') as f:
                    json.dump(batch_train_losses, f)
                
                # perform validation after each batch
                print('evaluating...')
                y_pred = run_softmax_eval(model, val_loader, cxr_labels, cxr_pair_template)
                
                roc_auc_all = []
                for i in range(y_pred.shape[-1]):
                    _, _, _, roc_auc = plot_roc(y_pred[:, i], y_true[:, i], '')
                    roc_auc_all.append(roc_auc)
                
                # save current batch's validation AUC
                avg_roc_auc = np.mean(roc_auc_all)
                batch_val_roc_auc.append(avg_roc_auc)
                batch_val_roc_auc_path = os.path.join(model_save_dir, "batch_val_roc_auc.json")
                with open(batch_val_roc_auc_path, 'w') as f:
                    json.dump(batch_val_roc_auc, f)

                # Report metrics every `report_freq` batch
                if (batch_ct % report_freq) == 0:
                    train_log(running_loss / report_freq, example_ct, epoch)
                    running_loss = 0.0
                
                # save if AUC is higher than best-so-far
                if avg_roc_auc > best_roc_auc:
                    checkpoint_name = f"checkpoint_batch_{batch_ct}_batchsize_{config.batch_size}_auc_{avg_roc_auc}.pt"
                    model_path = os.path.join(model_save_dir, checkpoint_name)
                    print("Saved checkpoint to: ", model_path)
                    save(model, model_path)
                    best_roc_auc = avg_roc_auc
                    

if __name__ == "__main__":
    experiment_pipeline(
        cxr_labels,
        cxr_pair_template,
        config
    )