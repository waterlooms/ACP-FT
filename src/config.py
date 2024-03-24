import os
import numpy as np
import torch
import logging


class Config(object):
	def __init__(self, dataset_name = 'ACP-MLC'):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#		self.esm_model_name = 'facebook/esm2_t6_8M_UR50D'
		self.esm_model_name = 'facebook/esm2_t30_150M_UR50D'
#		self.esm_model_name = 'facebook/esm2_t33_650M_UR50D'
		if dataset_name == 'ACP-MLC':
			self.all_labels = ['Colon','Breast','Cervix','Lung','Skin','Prostate','Blood']
		elif dataset_name == 'ACPScanner':
			self.all_labels = ['Blood', 'Breast', 'Cervical', 'Colon', 'Liver', 'Histiocyte', 'Lung', 'Myeloma', 'Prostate', 'Oth']
		self.label2id = {self.all_labels[i]: i for i in range(len(self.all_labels))}
		self.num_labels = len(self.all_labels)

	def convert_label(self, title):
		if title == 'Neg':
			return np.zeros(self.num_labels)
		labels = [self.label2id[x] for x in title.split(',')]
		one_hot_label = np.zeros(self.num_labels)
		for x in labels:
			one_hot_label[x] = 1
		return one_hot_label
	
		
class DataConfig(Config):
	def __init__(self, name, dataset_name, data_dir):
		super(DataConfig, self).__init__(dataset_name)
		self.name = name
		self.data_dir = data_dir
		# Input restriction
		self.max_length = 126

class ModelConfig(Config):
	def __init__(self, dataset_name):
		super(ModelConfig, self).__init__(dataset_name)
		print(f'Run on {self.device}')		
		self.dropout_rate = 0.5
		# ESM configs
		if self.esm_model_name == 'facebook/esm2_t6_8M_UR50D':
			self.esm_layer = 6
			self.esm_dimension = 320
		elif self.esm_model_name == 'facebook/esm2_t12_35M_UR50D':
			self.esm_layer = 12
			self.esm_dimension = 480
		elif self.esm_model_name == 'facebook/esm2_t30_150M_UR50D':
			self.esm_layer = 30
			self.esm_dimension = 640
		elif self.esm_model_name == 'facebook/esm2_t33_650M_UR50D':
			self.esm_layer = 33
			self.esm_dimension = 1280
			

class TrainerConfig(ModelConfig):
	def __init__(self, dataset_name, dataset_idx, str_time):
		super(TrainerConfig, self).__init__(dataset_name)

		self.batch_size = 16
		self.total_epoch, self.patience = 500, 40
		self.total_rounds = 5
		self.seed = 258

		self.current_path = os.getcwd()
		_dataset_folder = f'{self.current_path}/ACP_datasets'
		_tmp_folder = f'{self.current_path}/tmp'

		# dataset folder
		self.dataset_name = dataset_name
		if self.dataset_name == 'ACP-MLC':
			self.train_dir = f'{_dataset_folder}/ACP-MLC-10fold/train_{dataset_idx}.fasta'
			self.test_dir = f'{_dataset_folder}/ACP-MLC-10fold/test_{dataset_idx}.fasta'
			self.test_file = f'test_{dataset_idx}'
		elif self.dataset_name == 'ACPScanner':
			if dataset_idx == 0:
				self.train_dir = f'{_dataset_folder}/ACPScanner/train.fasta'
				self.test_dir = f'{_dataset_folder}/ACPScanner/test.fasta'
				self.test_file = f'test'
			else:
				self.train_dir = f'{_dataset_folder}/ACPScanner-10fold/train_{dataset_idx}.fasta'
				self.test_dir = f'{_dataset_folder}/ACPScanner-10fold/test_{dataset_idx}.fasta'
				self.test_file = f'test_{dataset_idx}'

		# tmp folder
		os.makedirs(_tmp_folder, exist_ok=True)
		self.save_dir = f'{_tmp_folder}/best_model/'
		os.makedirs(self.save_dir, exist_ok=True)
		_log_folder = f'{_tmp_folder}/logs'
		os.makedirs(_log_folder, exist_ok=True)
		self.loss_graph = f'{_tmp_folder}/loss.png'
		self.metric_graph = f'{_tmp_folder}/metric.png'

		# result folder
		self.result_folder = f'results/{str_time}'
		os.makedirs(self.result_folder, exist_ok=True)
		# log file
		self.log_dir = f'{self.result_folder}/run.log'
		self.logger = self.get_logger()
		# step table
		self.step_table = f'{self.result_folder}/steps.csv'
		self.column_names = ['Test_file', 'Rounds', 'Epoch', 'Train_loss', 'Val_loss', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1score', 'MCC']
		self.show_metric = ['AUROC']

	def get_logger(self):
		if 'Mylogger' in logging.Logger.manager.loggerDict:
			return logging.Logger.manager.loggerDict['Mylogger']
		else:
			return self.init_logger()
	
	def init_logger(self):
		logger = logging.getLogger('Mylogger')
		logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
		# Stream Handler
		handler = logging.StreamHandler()
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		# File Handler
		handler = logging.FileHandler(self.log_dir)
		handler.setFormatter(formatter)
		logger.addHandler(handler)
		# Print Configuration
		logger.info("### Print the current configuration to a log file ")
		for key, value in self.__dict__.items():
			logger.info(f"###  {key} = {value}")
		return logger