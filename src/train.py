import torch
from config import *
from torch import nn
import config
import wandb
from tqdm import tqdm, trange
from model import BertClassifier
from transformers import BertTokenizer, AdamW , get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F


class DoubleLoss(nn.Module):
    def __init__(self):
        super(DoubleLoss, self).__init__()
        self.bce_loss = nn.CrossEntropyLoss()
        self.kl_divloss = nn.KLDivLoss(reduction="batchmean")
        
        
    def kl_divergence(self, p, q, temperature=1.0):
        p = F.softmax(p / temperature, dim=0)
        q = F.softmax(q / temperature, dim=0)
        p = torch.clamp(p, 1e-10, 1.0).requires_grad_()
        q = torch.clamp(q, 1e-10, 1.0).requires_grad_()

        # Tính KL-divergence
        kl = F.kl_div(torch.log(p), q, reduction='sum')

        return kl
    
    def forward(self, output_bert, target_bert, output_mlp, target_mlp):
        
        # KL-divergence loss between the outputs of the two BERT models
        kl_loss = self.kl_divergence(output_bert, target_bert)
        
        # BCEWithLogitsLoss for the additional MLP
        bce_loss = self.bce_loss(output_mlp, target_mlp)

        return kl_loss + bce_loss


class Trainer:
    def __init__(self, tokenizer, train_dataset, val_dataset, type_data, 
                 strategy='adapter', weight_decay=0, learning_rate=1e-5,
                 warmup_steps=0, estimated_stepping_batches=32, adam_epsilon=1e-10):
        self.type_data = type_data
        self.model = BertClassifier(num_labels = 1)
        self.gpu_present = torch.cuda.is_available()
        
        self.strategy = strategy
        self.weight_decay = weight_decay
        
        self.trainable_params_count = 0
        self.learning_rate = learning_rate
        
        self.warmup_steps = warmup_steps
        self.estimated_stepping_batches = estimated_stepping_batches
        
        self.adam_epsilon = adam_epsilon
        
        wandb.init(project="Bert-Adapter-For-ED")

        if self.gpu_present:
            self.model = self.model.cuda()

        tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.train_dataset = self.type_data(train_dataset, tokenizer)
        self.val_dataset = self.type_data(val_dataset, tokenizer) 
        self.loss_fct = nn.DoubleLoss()


    def configure_optimizers(self):
        layers = ["adapter", "LayerNorm"]
        params = [p for n, p in self.model.named_parameters() \
                        if any([(nd in n) for nd in layers])]
        
        self.optimizer = AdamW(params, lr=config.LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps = int(0.1 * len(self.train_dataset))*config.EPOCHS
        )

    def configure_optimizers(self):
        model = self.model
        optimizer_grouped_parameters = []
        if self.strategy == 'full-finetuning':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        elif self.strategy == 'adapter':
            no_decay = ["adapter.proj_up.bias", "adapter.proj_down.bias", "LayerNorm"]
            cls_bias = ['cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias']
            cls_weight = ['cls.seq_relationship.weight', 'cls.predicions.transform.dense.weight', 'cls.predictions.decoder.weight']
            layers = ["adapter.proj_up.weight", "adapter.proj_down.weight"]
            layers.extend(cls_weight)
            no_decay.extend(cls_bias)
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        # count the total no. of trainable params
        for group in optimizer_grouped_parameters:
            for param in group["params"]:
                self.trainable_params_count += param.numel()
        print(f'Total Trainable params: {self.trainable_params_count}')

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]
    


    def compute_loss(self, output, labels):
        return self.loss_fct(output, labels)


    def train(self):
        # Setting Up DataLoaders and Optimizars
        trainloader = self.get_train_dataloader()
        valloader = self.get_val_dataloader()
        self.configure_optimizers()

        # Training Loop Starts Here
        for e in trange(config.EPOCHS):
            train_loss = 0.0
            # Training Step
            for batch in tqdm(trainloader):
                inputs, labels = batch
                if self.gpu_present:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                train_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Validation Step
            valid_loss += 0.0
            with torch.no_grad():
                for batch in tqdm(valloader):
                    inputs, labels = batch['sentence'], batch['labels']
                    if self.gpu_present:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    outputs = self.model(**inputs)
                    loss = self.compute_loss(outputs, labels)
                    valid_loss += loss.item()
            
            wandb.log({
                'epoch': e,
                'train_loss': train_loss/len(trainloader),
                'val_loss': valid_loss/len(valloader)
            })
            print(f'Epoch {e}\t\tTraining Loss: {train_loss/len(trainloader)}\t\tValidation Loss: {valid_loss/len(valloader)}')
        wandb.finish()

if __name__=='__main__':
    Trainer().train()