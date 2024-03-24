from pyteomics import fasta
from transformers import AutoTokenizer

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
	
	def __init__(self, config):
		print(f'Creating {config.name}')
		self.max_length = config.max_length
		self.sequences, self.labels = [], []
		if config.data_dir.endswith('.fasta'):
			dataset = fasta.read(config.data_dir)
			for x in dataset:
				title, seq = x.description, x.sequence
				if len(seq) > self.max_length:
					print(seq)
					continue
				one_hot_label = config.convert_label(title)
				self.sequences.append(seq)
				self.labels.append(one_hot_label)
		self.tokenizer = AutoTokenizer.from_pretrained(config.esm_model_name)
		print('Number of items:', self.__len__())
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		
		inputs = self.tokenizer(
			text=self.sequences[idx],
			max_length=self.max_length + 2,
			padding='max_length',
			return_tensors="pt",
		)
		ret = {
			'input_ids': inputs['input_ids'][0],
			'attention_mask': inputs['attention_mask'][0],
			'labels': self.labels[idx]
		}
		return ret


if __name__ == '__main__':
	from config import DataConfig
	'''
	TraindataConfig = DataConfig(
		name='Train set', 
		data_dir = 'ACP_datasets/ACP-MLC-10fold/train_1.fasta',
		dataset_name='ACP-MLC'
	)
	TraindataConfig = DataConfig(
		name='Train set', 
		data_dir = 'ACP_datasets/ACPScanner/train.fasta',
		dataset_name='ACPScanner'
	)
	'''
	TraindataConfig = DataConfig(
		name='Train set', 
		data_dir = 'ACP_datasets/ACPScanner-10fold/train_1.fasta',
		dataset_name='ACPScanner'
	)
	train_dataset = MyDataset(TraindataConfig)
	train_loader = DataLoader(
		train_dataset, 
		batch_size=8, 
		shuffle=False, 
#		drop_last=True,
	)
	for batch in train_loader:
		print(batch)
