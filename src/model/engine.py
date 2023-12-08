import torch
import torch.nn as nn
from .bert import BertModel
from .roberta import RobertaModel
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup


class BertClassifier(nn.Module):
    def __init__(self, model_path, tokenizer, num_labels):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bert_model = BertModel.from_pretrained(model_path)
        
        self.gpu_present = torch.cuda.is_available()
        
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        
        # Chỉnh sửa ở đây concat 2 entities lại thành 1 trả về concat_token
        
        
        cls_token = self.bert_model(x).last_hidden_state[0, 0, :]  # Sửa lỗi ở đây, để lấy tất cả các instance trong batch
        
        classifier = self.classifier(cls_token)
        output_probs = self.softmax(classifier)
        return cls_token, output_probs


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer_grouped_parameters = []
        if self.hparams.strategy == 'full-finetuning':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        elif self.hparams.strategy == 'adapter':
            no_decay = ["adapter.proj_up.bias", "adapter.proj_down.bias", "LayerNorm"]
            cls_bias = ['classifier.bias']
            cls_weight = ['classifier.weight']
            layers = ["adapter.proj_up.weight", "adapter.proj_down.weight"]
            layers.extend(cls_weight)
            no_decay.extend(cls_bias)
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": self.hparams.weight_decay,
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

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    
class RobertaClassifier(nn.Module):
    def __init__(self, model_path, tokenizer, num_labels):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bert_model = RobertaModel.from_pretrained(model_path)
        
        self.gpu_present = torch.cuda.is_available()
        
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        
        # Chỉnh sửa ở đây concat 2 entities lại thành 1 trả về concat_token
        
        
        cls_token = self.bert_model(x).last_hidden_state[0, 0, :]  # Sửa lỗi ở đây, để lấy tất cả các instance trong batch
        
        classifier = self.classifier(cls_token)
        output_probs = self.softmax(classifier)
        return cls_token, output_probs

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        optimizer_grouped_parameters = []
        if self.hparams.strategy == 'full-finetuning':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        elif self.hparams.strategy == 'adapter':
            no_decay = ["adapter.proj_up.bias", "adapter.proj_down.bias", "LayerNorm"]
            cls_bias = ['classifier.bias']
            cls_weight = ['classifier.weight']
            layers = ["adapter.proj_up.weight", "adapter.proj_down.weight"]     
            layers.extend(cls_weight)
            no_decay.extend(cls_bias)
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if any([nd in n for nd in layers])],
                    "weight_decay": self.hparams.weight_decay,
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

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]