import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import torch.nn as nn

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from transformers import AdamW, get_linear_schedule_with_warmup


class DLFramework(torch.nn.Module):
    def __init__(self, model, loss, config, device):

        super().__init__()

        # Set model
        self.model = model

        # Set loss
        self.loss = loss

        # Set training batch size
        self.batch_size = config['batch_size']

        # Set number of epochs
        self.epochs = config['epochs']

        # Set gradient clipping - True/False
        self.gradient_clipping = config['gradient_clipping']

        # Set optimizer
        self.optimizer = AdamW(self.model.parameters(),
                               lr=5e-5,    # Default learning rate
                               eps=1e-8    # Default epsilon value
                               )

        # Set scheduler
        # Total number of training steps
        total_steps = self.batch_size * self.epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)


        # Set data transformation function -- remove
        #self.input_transformation = input_transformation

        # Set softmax function
        self.softmax = nn.Softmax(dim=1)

        # Set device (CPU/GPU)
        self.device = device



        print('Train module initialized!')
      

    def train(self, train_data, epoch=0): 
        """ Trains model for 1 epoch. """

        # Setting model in train mode
        self.model.train()

        # Initializing data loader
        train_loader = DataLoader(dataset=train_data, 
                                  batch_size=self.batch_size,
                                  shuffle=True)

        # Epoch loss
        epoch_loss = 0.0

        # Batch counter
        batch_counter = 0

        # TODO - Confidence + correctness dict per epoch
        epoch_stats = {}

        # Epoch run
        for batch_id, batch in enumerate(train_loader):

            # Increment batch counter
            batch_counter += 1

            # Set batch timer
            t_batch = time.time() 

            # Extract data
            batch_ids = batch[0]
            input_ids = batch[1][0].to(self.device)
            attention_masks = batch[1][1].to(self.device)
            token_type_ids = batch[1][2].to(self.device)
            target_ids = batch[1][3].to(self.device)
            ner_types = batch[1][4].to(self.device)
            batch_labels = batch[2].to(self.device)    

            # FORWARD PASS

            # Set gradients to zero 
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model.forward(input_ids, attention_masks, token_type_ids, target_ids, ner_types)

            # TODO - Dodati softmax za logite
            # TODO - Izračunati p(y|x) - confidence
            # TODO - updateati confidence dict

            # Apply loss function
            batch_loss = self.loss(outputs, batch_labels)
            epoch_loss += batch_loss.item()


            # BACKWARD PASS
            
            # Backpropagation
            batch_loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            if self.gradient_clipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update model paramaters
            self.optimizer.step()

            # Print the loss values and time elapsed for every 20 batches
            if (batch_id % 2 == 0 and batch_id != 0) or (batch_id == len(train_loader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t_batch

                # Print training results
                print(f"{epoch + 1:^7} | {batch_id:^7} | {batch_loss:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

        # Schedule learning rate - after epoch is done
        if self.scheduler:
            self.scheduler.step()

        # Caluclate epoch metrics and mean loss
        avg_epoch_loss = epoch_loss / batch_counter

        return avg_epoch_loss, epoch_stats

    def eval(self, val_data):
  
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.model.eval()

        # Initializing data loader
        val_loader = DataLoader(dataset=val_data, 
                            batch_size=len(val_data),
                            shuffle=False)

        # Tracking variables
        val_f1 = []
        val_loss = []

        # True and predicted labels list
        y_true = []
        y_pred = []

        # For each batch in our validation set...
        for batch in val_loader:

            # Set batch timer
            t_batch = time.time()

            # Extract data
            batch_ids = batch[0]
            input_ids = batch[1][0].to(self.device)
            attention_masks = batch[1][1].to(self.device)
            token_type_ids = batch[1][2].to(self.device)
            target_ids = batch[1][3].to(self.device)
            ner_types = batch[1][4].to(self.device)
            batch_labels = batch[2].to(self.device) 


            # Compute logits
            with torch.no_grad():
                logits =  self.model.forward(input_ids, attention_masks, token_type_ids, target_ids, ner_types)

            # Compute loss
            loss = self.loss(logits, batch_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(self.softmax(logits), dim=1).flatten()

            # Append true and predicted labels to list
            y_true.extend(batch_labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
        

        # Compute the f1 and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_f1 = f1_score(y_true, y_pred, average = 'macro')
        
        return val_loss, val_f1, y_true, y_pred


    def run(self, train_data, val_data):
        
        # Training stats - confidence + correctness
        training_stats = {}

        # Epoch stats (train_loss, val_loss, val_f1)
        model_stats = {'train_loss':[],
                       'val_loss': [],
                       'val_f1':[]
                       }

        # Epoch clf report
        clf_reports = []

        for epoch in range(self.epochs):

            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val F1':^9} | {'Elapsed':^9}")
            print("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch = time.time()

           # Train
            train_loss, epoch_stats = self.train(train_data)
            model_stats['train_loss'].append(train_loss)

            print("-"*70)

            # Eval
            val_loss, val_f1, y_true, y_pred = self.eval(val_data)
            model_stats['val_loss'].append(val_loss)
            model_stats['val_f1'].append(val_f1)


            # Epoch time
            time_elapsed = time.time() - t0_epoch

            # Print results
            print(f"{epoch + 1:^7} | {'-':^7} | {train_loss:^12.6f} | {val_loss:^10.6f} | {val_f1:^10.6f} | {time_elapsed:^9.2f}")
            print("-"*70)
            print("\n")

            clf_reports.append(self.get_clf_report(y_true, y_pred))
            self.get_conf_matrix(y_true, y_pred)

        print("Done!")

        return model_stats, clf_reports
  


    def get_module_config(self):

        module_config = {'model': str(self.model),
                         'loss': str(self.loss),
                         'optimizer': str(self.optimizer),
                         'scheduler': str(self.scheduler),
                         'scheduler_params': str(self.scheduler.__dict__),
                         'gradient_clipping': str(self.gradient_clipping),
                         'input_transformation': str(self.input_transformation.__dict__),
                         'batch_size': str(self.batch_size),
                         'epochs': str(self.epochs)
                         }

        return module_config


    def get_clf_report(self, y_true, y_pred):
        print('-------------------------------------------------------')
        print('Classification report:\n-------------------------------------------------------')

        print(classification_report(y_true, y_pred, target_names=['0','1','2']))

        print('-------------------------------------------------------')

        clf_report = classification_report(y_true, y_pred, target_names=['0','1','2'], output_dict=True)
        clf_df = pd.DataFrame(clf_report).transpose()

        return clf_df

    
    def get_conf_matrix(self, y_true, y_pred):
        # Confusion matrix

        print('-------------------------------------------------------')
        print('Confusion matrix:\n-------------------------------------------------------')

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        plt.rcParams["figure.figsize"] = (8,5.5)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
        disp.plot()
        plt.show()


    