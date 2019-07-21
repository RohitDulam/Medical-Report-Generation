import json
import torch
import os
import numpy
from torch.utils.data import Dataset, DataLoader
from operator import itemgetter

data_dir = '/'

def get_length(strings, ns, nw):
	ls = []
	count_words = []
	for i in strings:
		t = '<start> ' + i + ' <end>'
		count_words.append(len(t.split()))
		if len(t.split()) < nw:
			w = nw - len(t.split())
			z = t + w * ' <pad>'
			ls.append(z)
		else:
			print(nw, len(t.split()))
			ls.append(t)
	# Filling the number of sentences problem.
	if len(ls) < ns:
		s = ns - len(ls)
		for _ in range(s):
			ls.append(nw * '<pad> ')

	return ls, len(strings), max(count_words)

def get_tensor(strings, vocab, ns, nw):
	create_batch = []
	length = len(vocab)
	for i in strings:
		inter_batch = []
		x = i.split()
		for j in x:
			#inter_batch.append(vocab[j])
			inter_batch.append(one_hot(vocab[j], length))
		inter_batch = [v for v in inter_batch]
		#print(len(inter_batch[0]))
		inter_batch = torch.cat(inter_batch)
		#create_batch.append(inter_batch)
		create_batch.append(inter_batch)
	create_batch = [v for v in create_batch]
	#create_batch = torch.cat(create_batch).view(13, 36, 3266)
	#return torch.tensor(create_batch)
	#return create_batch
	#print(torch.cat(create_batch).size())
	return torch.cat(create_batch).view(ns, nw, length)

def one_hot(index, length):
	ls = torch.zeros(length)
	ls[index - 1] = 1
	return ls

def get_images(img_dir):
	img = cv2.imread(img_dir)
	img = cv2.resize(img, (224, 224))
	return img

def tokenization(data_dir):
	with open(data_dir) as f:
		data = json.load(f)
	data = [v for v in data.values()]
	details = []
	for i in data:
		h = i['fnd']
		g = i['imp']
		for j in h:
			details.append(j)
		for j in g:
			details.append(j)

	count = 3 
	ls = []
	d = {'<start>' : 0,'<end>' : 1,'<pad>' : 2}
	ls.append('<start>')
	ls.append('<end>')
	ls.append('<pad>')
	for i in details:
		x = i.split()
		for j in x:
			if j in ls:
				continue
			else:
				ls.append(j)
				d[j] = count
				count += 1

	return d

class Loader(Dataset):
	"""docstring for Loader"""
	def __init__(self, data_dir):
		super(Loader, self).__init__()
		with open(data_dir) as f:
			data = json.load(f)
		self.data = [v for v in data.values()]
		self.vocab = tokenization(data_dir)
		self.count_s_i = 0
		self.count_s_f = 0
		self.count_w_i = 0
		self.count_w_f = 0
		for i in self.data:
			name_f = i['fnd']
			name_i = i['imp']

			if len(name_f) > self.count_s_f:
				self.count_s_f = len(name_f)

			for j in name_f:
				x = j.split()
				if (len(x) + 2) > self.count_w_f:
					self.count_w_f = len(x) + 2

			if len(name_i) > self.count_s_i:
				self.count_s_i = len(name_i)

			for j in name_i:
				x = j.split()
				if (len(x) + 2) > self.count_w_i:
					self.count_w_i = len(x) + 2

	def __getitem__(self, i):
		#img_frnt = self.data[i]['img_frt']
		#img_sd = self.data[i]['img_sd']
		fnd = self.data[i]['fnd']
		imp = self.data[i]['imp']
		fnd_final, fnd_length, fnd_word_count = get_length(fnd, self.count_s_f, self.count_w_f)
		imp_final, imp_length, imp_word_count = get_length(imp, self.count_s_i, self.count_w_i)
		#print(fnd_final)
		#print(imp_final)
		fnd_final = get_tensor(fnd_final, self.vocab, self.count_s_f, self.count_w_f)
		imp_final = get_tensor(imp_final, self.vocab, self.count_s_i, self.count_w_i)
		## The below 2 lines should be uncommented when trying to run on local machine. The image directory should point to the directory which contains all the images. 
		#frnt_dir = self.img_dir + img_frnt.split('/')[-1] # The second term gives us the name of the image. 
		#sd_dir = self.img_dir + img_sd.split('/')[-1]
		# use frnt_dir and sd_dir to get the images
		# Except for reading the image, everything else is set up I feel. 
		img_frnt = imgs = torch.rand(224, 224, 3).cuda() # call get_images() method once the data dir is set. 
		return [img_frnt, img_frnt, fnd_final, imp_final, fnd_length, fnd_word_count, imp_length, imp_word_count] # Add img_sd here.

	def __len__(self):
		return len(self.data)


'''if __name__ == '__main__':
	dataset_testing = Loader('id_data_all_sides.json') # Add img_dir here and also in the constructor of the above class while running on local machine. 
	dataloader = DataLoader(dataset_testing, batch_size = 6, shuffle = False, num_workers = 0)
	for x, y in enumerate(dataloader):
		#print(y[1])
		#m, sort = torch.sort(y[-4])
		#print(m, sort.numpy())
		#testing = sorted(y, key = itemgetter(4))
		#print(testing[1])
		#frnt_img = list(y[0])
		#print(frnt_img)
		#print([x for _,x in sorted(zip(sort.numpy(),frnt_img))])
		#m = y[6]
		# = y[2]
		#print(m)
		break'''

		