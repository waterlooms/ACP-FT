import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import datetime
import random


from data import MyDataset
from runner import Runner
from model import Linear_probing, Full_finetune
from config import DataConfig, TrainerConfig
import util

class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.setup_seed(config.seed)
        #initialize step table
        with open(self.config.step_table, 'w') as fw:
            fw.write(','.join(self.config.column_names))
            fw.write('\n')
        self.current_round = None
        self.df_template = util.convert_fasta_df(self.config.test_dir).drop(columns=['label'])
#      self.methods = ['FT', 'LP-FT-1', 'LP-FT-2', 'EH-FT-1', 'EH-FT-2']
        self.methods = ['FT', 'LP-FT-1', 'LP-FT-2']
        self.results = {m:[] for m in self.methods}

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
 
    def _draw_loss(self, train_loss, val_loss):
        plt.figure()
        plt.plot(train_loss, label = 'train loss')
        plt.plot(val_loss, label = 'val loss')
        plt.title(f'Loss: {self.config.test_file}_{self.current_round}')
        plt.legend()
        plt.savefig(self.config.loss_graph)
        plt.close()


    def _draw_metric(self, metrics, name_list):
        plt.figure()
        for name in name_list:
            y = metrics[name]
            plt.plot(y, label = name)
        plt.title(f'Metric: {self.config.test_file}_{self.current_round}')
        plt.legend()
        plt.savefig(self.config.metric_graph)
        plt.close()

    def record_step(self, epoch, train_loss, val_loss, labels, outputs):
        losses = {
            'Test_file': self.config.test_file,
            'Rounds': self.current_round,
            'Epoch': epoch + 1,
            'Train_loss': train_loss,
            'Val_loss': val_loss,
        }
        metrics, _ = util.compute_metric(labels, outputs)
        colomns = {**losses, **metrics}
        with open(self.config.step_table, 'a') as fw:
            write_list = []
            for x in self.config.column_names:
                v = colomns[x]
                if type(v) is str:
                    write_list.append(v)
                else:
                    write_list.append(str(round(v, 4)))
            fw.write(','.join(write_list))
            fw.write('\n')

    def draw(self):
        df = pd.read_csv(self.config.step_table)
        df_show = df[(df['Test_file'] == self.config.test_file) & (df['Rounds'] == self.current_round)]
        metrics = df_show[self.config.show_metric]
        self._draw_metric(metrics=metrics, name_list=self.config.show_metric)
        self._draw_loss(train_loss=df_show['Train_loss'], val_loss=df_show['Val_loss'])
    
    
    def train_model(self, train_loader, val_loader, runner):
        best_loss, best_idx = 1e10, -1
        counter = 0
        for epoch in range(self.config.total_epoch):
            train_loss = runner.train(train_loader)
            val_loss, outputs, labels = runner.val(val_loader)
            self.record_step(epoch, train_loss, val_loss, labels, outputs)
            self.logger.info(
                f"{self.config.test_file} Round:{self.current_round}-{runner.name} "
                f"Epoch: [{epoch+1}/{self.config.total_epoch}] "
                f"Train loss: {(train_loss):.3f} "
                f"Val loss: {(val_loss):.3f}"
            )
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_idx = epoch
#            runner.save(self.config.save_dir + f'{self.config.test_file}_{self.current_round}_{runner.name}.pth')
                counter = 0
            else:
                counter += 1
                if counter > self.config.patience:
                    break
            if epoch % 5 == 4:
                self.draw()
        self.logger.info(f"Best model is at epoch {best_idx}: {round(best_loss,4)}")


    def add_results(self, runner):
        testConfig = DataConfig(
            name='Test set', 
            dataset_name=self.config.dataset_name,
            data_dir=self.config.test_dir,
        )
        test_dataset = MyDataset(testConfig)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        _, outputs, labels = runner.val(test_loader)
        if self.current_round == 0:
            self.results[runner.name].append(labels)
        self.results[runner.name].append(outputs)
        self.logger.info(f"Round:{self.current_round}-{runner.name}")
        self.log_result(labels, outputs)
        df = self.df_template.copy()
        for i in range(len(self.config.all_labels)):
            column_name = self.config.all_labels[i]
            df[column_name] = outputs[:,i]
        df = df.round(4)
        folder = f'{self.config.result_folder}/{self.config.test_file}/{runner.name}'
        os.makedirs(folder, exist_ok=True)
        df.to_csv(f'{folder}/{self.current_round}.csv', index=False)


    
    def train(self):
        for i in range(self.config.total_rounds):
            self.current_round = i
            trainConfig = DataConfig(
                name='Train set', 
                dataset_name=self.config.dataset_name,
                data_dir=self.config.train_dir
            )
            train_dataset = MyDataset(trainConfig)
            train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # FT #
#            runner = Runner(self.config, Full_finetune, name='FT')
#            self.train_model(train_loader, val_loader, runner)
#            self.add_results(runner)

            # LP-FT #
            runner = Runner(self.config, Linear_probing, name='LP-FT-1')
            self.train_model(train_loader, val_loader, runner)
            self.add_results(runner)
            trained_lp = runner.model.classifier

            runner = Runner(self.config, Full_finetune, name='LP-FT-2')
#            runner.lr *= 0.1
            runner.initialize_linear(trained_lp)
            self.train_model(train_loader, val_loader, runner)
            self.add_results(runner)
        self.write_result()

    def write_result(self):
        for name in self.methods:
            result = self.results[name]
            labels = result[0]
            outputs = np.mean(result[1:], axis=0)
            self.logger.info(f"Final prediction-{name}")
            self.log_result(labels, outputs)
            df = self.df_template.copy()
            for i in range(len(self.config.all_labels)):
                column_name = self.config.all_labels[i]
                df[column_name] = outputs[:,i]
            df = df.round(4)
            df.to_csv(f'{self.config.result_folder}/{self.config.test_file}/{name}/average.csv', index=False)
    
    def log_result(self, labels, outputs):
        metrics_mean, metrics_label = util.compute_metric(labels, outputs)
        for x in metrics_mean:
            logging_str = f"{x}: {round(metrics_mean[x], 4)}-"
            if x in metrics_label:
                logging_str += " ".join([str(round(num, 4)) for num in metrics_label[x]])
                self.logger.info(logging_str)
    

def train_model_1():
    str_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    for i in range(1, 11):
        trainerConfig = TrainerConfig(dataset_name='ACP-MLC', dataset_idx=i, str_time = str_time)
        trainer = Trainer(trainerConfig)
        trainer.train()
    
def train_model_2():
    str_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    trainerConfig = TrainerConfig(dataset_name = 'ACPScanner', dataset_idx=0, str_time = str_time)
    trainer = Trainer(trainerConfig)
    trainer.train()

def train_model_3():
    str_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    for i in range(1, 11):
        trainerConfig = TrainerConfig(dataset_name='ACPScanner', dataset_idx=i, str_time = str_time)
        trainer = Trainer(trainerConfig)
        trainer.train()


if __name__ == "__main__":
    train_model_2()