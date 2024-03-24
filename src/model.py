import torch
import torch.nn as nn
from transformers import EsmModel

class Base(nn.Module):

    def __init__(self, config):
        super(Base, self).__init__()
        self.esm_model = EsmModel.from_pretrained(
            config.esm_model_name,
        )
        self.esm_layer = config.esm_layer
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()


class ClassificationHead(nn.Module):

    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.esm_dimension, config.esm_dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.esm_dimension, config.num_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        

class Linear_probing(Base):

    def __init__(self, config):
        super(Linear_probing, self).__init__(config)
        '''Freeze ESM'''
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.classifier = ClassificationHead(config)
        self.lr = 1e-4
        self.adversarial = False
    
    def forward(self, x, y=None):
        emb = self.esm_model(**x).last_hidden_state
        pred_emb = emb[:, 0]
        prediction_res = self.classifier(pred_emb)
        if y is not None:
            loss = self.criterion(prediction_res, y)
            return loss, self.sigmoid(prediction_res)
        else:
            return self.sigmoid(prediction_res)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Full_finetune(Base):

    def __init__(self, config):
        super(Full_finetune, self).__init__(config)
        '''UnFreeze ESM'''
        for param in self.esm_model.parameters():
            param.requires_grad = True
        self.classifier = ClassificationHead(config)
        self.lr = 1e-5
        self.adversarial = True

    def forward(self, x, y=None):
        emb = self.esm_model(**x).last_hidden_state
        pred_emb = emb[:, 0]
        prediction_res = self.classifier(pred_emb)
        if y is not None:
            loss = self.criterion(prediction_res, y)
            return loss, self.sigmoid(prediction_res)
        else:
            return self.sigmoid(prediction_res)
    
