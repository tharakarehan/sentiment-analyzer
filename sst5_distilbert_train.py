#!usr/bin/env python3

# Import all libraries
import pandas as pd
import numpy as np
import re

# Huggingface transformers
import transformers
from transformers import BertModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn ,cuda
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#handling html data
from bs4 import BeautifulSoup

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

dataset_path = "sst_5_dataset.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = TensorBoardLogger(save_dir='logs/', name="sst5-Distil-bert")

sst5_df = pd.read_csv(dataset_path)
sst5_df['sentiment'] = sst5_df['sentiment']-1
train_df = sst5_df.loc[sst5_df['phase']=='train',['text','sentiment']]
test_df = sst5_df.loc[sst5_df['phase']=='test',['text','sentiment']]

x_tr = train_df['text'].tolist()
y_tr = train_df['sentiment'].tolist()
x_val = test_df['text'].tolist()
y_val = test_df['sentiment'].tolist()
x_val,x_test,y_val,y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

class DistBDataset(Dataset):

  def __init__(self, text, sentiment, tokenizer, max_len):
    self.tokenizer = tokenizer
    self.text = text
    self.label = sentiment
    self.max_len = max_len

  def __len__(self):
    return len(self.text)

  def __getitem__(self, item_idx):
    text = self.text[item_idx]
    label = self.label[item_idx]
    inputs = self.tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=self.max_len,
        padding= 'max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = torch.squeeze(inputs['input_ids'])
    attn_mask = torch.squeeze(inputs['attention_mask'])

    return {
        'input_ids' : input_ids,
        'attention_mask' : attn_mask,
        'label' : label
    }
    
class DistBDataModule(pl.LightningDataModule):
  
  def __init__(self, x_tr, y_tr, x_val, y_val, x_test, y_test, tokenizer, batch_size=16, max_token_len=200):
    super().__init__()
    self.tr_text = x_tr
    self.tr_label = y_tr
    self.val_text = x_val
    self.val_label = y_val
    self.test_text = x_test
    self.test_label = y_test
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_token_len = max_token_len

  def setup(self):
    self.train_dataset = DistBDataset(text=self.tr_text,
                                      sentiment=self.tr_label,
                                      tokenizer=self.tokenizer,
                                      max_len=self.max_token_len)
    self.val_dataset = DistBDataset(text=self.val_text,
                                    sentiment=self.val_label,
                                    tokenizer=self.tokenizer,
                                    max_len=self.max_token_len,)
    self.test_dataset = DistBDataset(text=self.test_text,
                                    sentiment=self.test_label,
                                    tokenizer=self.tokenizer,
                                    max_len=self.max_token_len,)
    
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size)

BERT_MODEL_NAME = "distilbert-base-uncased" # we will use the BERT base model(the smaller one)
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the parameters that will be use for training
N_EPOCHS = 30
BATCH_SIZE = 32
MAX_LEN = 300
LR = 2e-05

DisBdata_module = DistBDataModule(x_tr, y_tr, x_val, y_val, x_test, y_test, Bert_tokenizer, BATCH_SIZE, MAX_LEN)
DisBdata_module.setup()

class DisBClassifier(pl.LightningModule):

  def __init__(self, n_classes=3, setps_per_epoch=None, n_epochs=3, lr=2e-5):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.setps_per_epoch = setps_per_epoch
    self.n_epochs = n_epochs
    self.lr = lr
    self.criterior = nn.CrossEntropyLoss()

  def forward(self, inputs, attn_mask):
    output = self.bert(input_ids=inputs, attention_mask=attn_mask)
    output = self.classifier(output.pooler_output)
    return output

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']

    output = self(input_ids, attention_mask)
    loss = self.criterior(output, label)
    acc = accuracy(torch.argmax(output, dim=-1), label)
    self.log('train_loss', loss, prog_bar=True, logger=True)
    self.log('train_acc', acc, logger=True)
    logs = {'train_loss' : loss, 'train_acc' : acc}
    return {"loss" : loss,
            "_log" : logs,
            "prediction" : output,
            "label" : label}

  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']

    output = self(input_ids, attention_mask)
    loss = self.criterior(output, label)
    acc = accuracy(torch.argmax(output, dim=-1), label)
    self.log('val_loss', loss, prog_bar=True, logger=True)
    self.log('val_acc', acc, logger=True)
    logs = {'val_loss' : loss, 'val_acc' : acc}
    return {
        "loss" : loss,
        "log" : logs
    }

  def test_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']

    output = self(input_ids, attention_mask)
    loss = self.criterior(output, label)
    acc = accuracy(torch.argmax(output, dim=-1), label)
    self.log('test_loss', loss, prog_bar=True, logger=True)
    self.log('test_acc', acc, logger=True)
    logs = {'test_loss' : loss, 'test_acc' : acc}
    return {
        "loss" : loss,
        "log" : logs
    }

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr)
    warmup_steps = self.setps_per_epoch/3
    total_steps = self.setps_per_epoch*self.n_epochs - warmup_steps

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer], [scheduler]

#Initializing classifier model
steps_per_epoch = len(x_tr)//BATCH_SIZE
model = DisBClassifier(n_classes=3,setps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

# saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='DisB-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

early_stop_callback = EarlyStopping(
   monitor='val_acc',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='max'
)

# Instantiate the Model Trainer
trainer = pl.Trainer(max_epochs = N_EPOCHS , gpus = 1, callbacks=[checkpoint_callback, early_stop_callback], progress_bar_refresh_rate = 30, logger=logger)

trainer.fit(model, DisBdata_module)
#trainer.test(model, DisBdata_module)

model_path = checkpoint_callback.best_model_path
DisBmodel = DisBClassifier.load_from_checkpoint(model_path)
DisBmodel.eval()

question = "based on the following relationship between matthew s correlation coefficient mcc and chi square mcc is the pearson product moment correlation coefficient is it possible to conclude that by having imbalanced binary classification problem n and p df following mcc is significant mcc sqrt which is mcc when comparing two algorithms a b with trials of times if mean mcc a mcc a mean mcc b mcc b then a significantly outperforms b thanks in advance edit roc curves provide an overly optimistic view of the performance for imbalanced binary classification regarding threshold i m not a big fan of not using it as finally one have to decide for a threshold and quite frankly that person has no more information than me to decide upon hence providing pr or roc curves are just for the sake of circumventing the problem for publishing"

text_enc = Bert_tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length= MAX_LEN,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'      
    )
torch_out = DisBmodel(text_enc['input_ids'], text_enc['attention_mask'])


torch.onnx.export(DisBmodel,               # model being run
                  (text_enc['input_ids'], text_enc['attention_mask']),                         # model input (or a tuple for multiple inputs)
                  "sst5_distilbert.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

print("FINISH !!!")
