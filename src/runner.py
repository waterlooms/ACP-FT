import numpy as np
import torch
import torch.optim as optim
from model import Linear_probing, Full_finetune, FGM


class Runner:
    def __init__(self, config, Model, name):
        self.device = config.device
        self.model = Model(config)
        self.model.to(self.device)
        self.fgm = FGM(self.model)
        self.lr = self.model.lr
        self.adversarial = self.model.adversarial
        self.name = name

    def train(self, train_loader):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr = self.lr)
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [batch[item].to(self.device) for item in ['input_ids', 'attention_mask', 'labels']]
            inputs = {'input_ids':input_ids, 'attention_mask':attention_mask}
            optimizer.zero_grad()
            loss, outputs = self.model(x=inputs, y=labels)
            loss.backward()
            if self.adversarial:
                self.fgm.attack()
                loss_adv, outputs = self.model(x=inputs, y=labels)
                loss_adv.backward()
                self.fgm.restore()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        return train_loss

    def val(self, val_loader):
        self.model.eval()
        val_loss = 0
        outputs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [batch[item].to(self.device) for item in ['input_ids', 'attention_mask', 'labels']]
                inputs = {'input_ids':input_ids, 'attention_mask':attention_mask}
                loss, outputs = self.model(x=inputs, y=labels)
                val_loss += loss.item()
                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        val_loss /= len(val_loader)
        outputs_list, labels_list = np.vstack(outputs_list), np.vstack(labels_list)
        return val_loss, outputs_list, labels_list

    def save(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)

    def load(self, load_dir):
        self.model.load_state_dict(torch.load(load_dir))
    
    def print_params(self):
        for n, v in self.model.named_parameters():
            if v.requires_grad:
                print(n)
    
    def _test(self, test_loader):
        self.model.eval()
        val_loss = 0
        outputs_list, labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch.to(self.device)
                loss, outputs, labels = self.model(batch, mode='val')
                val_loss += loss.item()
                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        val_loss /= len(test_loader)
        outputs_list, labels_list = np.vstack(outputs_list), np.vstack(labels_list)
        print(labels_list.shape, outputs_list.shape)

    def initialize_linear(self, trained_lp):
        self.model.classifier = trained_lp

def test2():
    from config import ModelConfig, DataConfig
    dataset_name = 'ACPScanner'
    modelConfig = ModelConfig(dataset_name = 'ACPScanner')
    '''
    TraindataConfig = DataConfig(
        name='Train set', 
        data_dir = 'ACP_datasets/ACP-MLC-10fold/train_1.fasta',
        dataset_name='ACP-MLC'
    )
    
    TraindataConfig = DataConfig(
        name='Train set', 
        data_dir='ACP_datasets/ACPScanner/train.fasta',
        dataset_name=dataset_name
    )
    '''
    TraindataConfig = DataConfig(
        name='Train set', 
        data_dir='ACP_datasets/ACPScanner-10fold/train_1.fasta',
        dataset_name=dataset_name
    )
    from data import MyDataset
    from torch.utils.data import DataLoader
    train_dataset = MyDataset(TraindataConfig)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    name = 'Linear_probing'
    linear_probing = Runner(modelConfig, Linear_probing, name)
    linear_probing.print_params()
    train_loss = linear_probing.train(train_loader)

    name = 'Full_finetune'
    full_finetune = Runner(modelConfig, Full_finetune, name)
    full_finetune.initialize_linear(linear_probing.model.classifier)
    full_finetune.print_params()
    train_loss = full_finetune.train(train_loader)
    
if __name__ == '__main__':
    test2()