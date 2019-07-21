import torch
from torch.utils import data
from torch import nn
import torchvision
import numpy
import os
from xml.dom import minidom
from tqdm import tqdm
import cv2
from loader import Loader

img_size = 224
batch_size = 4
vocab_dim = 3266

params = {'batch_size' : batch_size, 'shuffle' : False, 'num_workers' : 1}
dataset = Loader('id_data_all_sides.json')

training_set = data.DataLoader(dataset, **params)

class Resnet(nn.Module):
	def __init__(self):
		super(Resnet, self).__init__()
		self.resnet_modules = torchvision.models.resnet50(pretrained = True).cuda()
		self.resnet_local = list(self.resnet_modules.children())[:-3]
		self.resnet_global = list(self.resnet_modules.children())[:-1]

	def forward(self, X):
		resn = nn.Sequential(*self.resnet_local).cuda()
		local_features = resn(X.view(batch_size, 3, img_size, img_size)) #(batch_size, output_channels = 1024, 14, 14) is the output shape.
		resn = nn.Sequential(*self.resnet_global)
		global_features = resn(X.view(batch_size, 3, img_size, img_size))# Add for getting global features. (batch_size, output_channels = 2048, 1, 1) is the output shape
		return local_features, global_features


class SentenceG(nn.Module):
	def __init__(self, vocab_dim, in_dim, out_dim):
		super(SentenceG, self).__init__()
		self.vocab_dim = vocab_dim
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.rnn = nn.LSTMCell(self.in_dim, self.out_dim).cuda()
		self.fc = nn.Linear(self.vocab_dim, self.in_dim).cuda()
		self.fc_ = nn.Linear(self.in_dim, self.out_dim).cuda()
		self.fcr = nn.Linear(2048, self.in_dim).cuda()
		self.fc2 = nn.Linear(self.out_dim, self.vocab_dim).cuda()

	def forward(self, inputs, teacher_forcing = True): # inputs = [images, sentences, number_of_sentences, number_of_words]
		images = inputs[0]
		images = torch.squeeze(images)

		if teacher_forcing:
			h = torch.zeros((batch_size, self.out_dim)).cuda()
			c = torch.zeros((batch_size, self.out_dim)).cuda()
			ns = inputs[2] # Change this accordingly
			nw = inputs[3] # Change this accordingly
			#print(ns, nw)
			#print(max(ns), max(nw))
			sentences_h = inputs[1]
			word_list = torch.zeros(batch_size, max(ns), max(nw), self.vocab_dim)
		#sort, indices = torch.sort(inputs[-2])
			for t in range(max(ns)): # change this accordingly. Maximum of all the input sentences. 
			#batch_size = sum([l > t for l in fnd_lengths_sorted])
				for u in range(1, max(nw)):# change this accordingly. Maximum of all the input words.
					#batch_size = sum([l > u for l in fnd_word_lengths_sorted])
					if t == 0:
						transformed = self.fcr(images)
						h, c = self.rnn(transformed, (h, c))
						word = self.fc2(h)
						word_list[:, 0, u, :] = word

					else:
						inp = torch.squeeze(sentences_h[:, t, u, :]).cuda()
						inp = self.fc(inp)
						h, c = self.rnn(inp, (h, c)) 
						word = self.fc2(h)  
						word_list[:, t, u, :] = word

		else:
			## Handle the inputs here.
			continue 
			#word_list = torch.zeros()
			#word = self.fcr(images)
			#h, c = self.rnn(word, (h, c))
			#word = self.fc2(h)
			#word_list[:, 0, u, :] = word
			## Break when an empty sentence is created. 
			#if word == :
				#break

		return word_list

class SentenceEncoder(nn.Module):
	"""docstring for SentenceEncoder"""
	def __init__(self, in_dim, out_dim):
		super(SentenceEncoder, self).__init__()
		## in_dim is 512
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.rnn = nn.LSTM(self.in_dim, self.out_dim, bidirectional = True).cuda()# Input size, hidden_size, num_layers and 

	def forward(self, inputs):
		h = torch.zeros(2, batch_size, self.out_dim).cuda()
		c = torch.zeros(2, batch_size, self.out_dim).cuda()
		output, hidden = self.rnn(inputs, (h,c))
		return hidden

class SentenceAttentionalDecoder(nn.Module):
	"""docstring for SentenceAttentionalDecoder"""
	def __init__(self, in_dim, out_dim, Encoder, SentenceG, Resnet):
		super(SentenceAttentionalDecoder, self).__init__()
		#self.linear_1 = nn.Linear()
		self.encoder = Encoder
		self.res = Resnet
		self.sentenceg = SentenceG
		self.s = nn.Linear(2048, 196,bias = False).cuda()
		self.v = nn.Linear(1024, 196,bias = False).cuda()
		self.a = nn.Linear(196 , 1,bias = False).cuda()
		self.ones = torch.ones(1, 196).cuda()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.LSTM1 = nn.LSTMCell(self.in_dim, self.out_dim).cuda()
		self.LSTM2 = nn.LSTMCell(self.out_dim, self.out_dim).cuda()
		self.f = nn.Linear(self.out_dim, vocab_dim).cuda()
		#self.LSTM = nn.LSTM(self.in_dim, self.out_dim, num_layers = 2).cuda()

	def forward(self, inputs, teacher_forcing):
		if teacher_forcing:
			continue
		else:
			impressions = self.sentenceg(inputs)
			## Break when an empty sentence is created. 
		#features = self.res()
		#seng = self.sentenceg(features, impressions)
		#fnd = inputs[2]
		#fnd = inputs[2]
		#fnd = inputs[2]
		#fnd_lengths = inputs[5]
		#fnd_lengths_sorted, indices = torch.sort(fnd_lengths)
		#fnd_word_lengths = inputs[6]
		#fnd_word_lengths_sorted, indices_w = torch.sort(fnd_word_lengths) 
		frnt_images = inputs[0]
		sentences = inputs[3]
		#print(sentences.size())
		sentences_f = inputs[2].cuda()
		#ns = inputs[2] # This 
		nsf = inputs[4]
		nwf = inputs[5]
		nsi = inputs[6]
		nwi = inputs[7]
		#nw = inputs[3]
		local_features, global_features = self.res(frnt_images)
		impressions = self.sentenceg([global_features, sentences, nsi, nwi]).cuda()
		#print(impressions[:, 0, :, :].permute(1, 0, 2).size())
		mummy = impressions.size()
		#print(mummy[0], mummy[1], mummy[2], mummy[3])
		word_list = torch.zeros(batch_size, max(nsf), max(nwf), vocab_dim).cuda()
		h = torch.zeros(batch_size, self.out_dim).cuda()
		c = torch.zeros(batch_size, self.out_dim).cuda()
		h1 = torch.zeros(batch_size, self.out_dim).cuda()
		c1 = torch.zeros(batch_size, self.out_dim).cuda()
		for t in range(max(nsf)): # change to max(fnd_lengths_sorted)
			if t == 0:
				sentenc, m = self.encoder(impressions[:, 0, :, :].permute(1, 0, 2))
				#print(m.size())
				sentenc = torch.unsqueeze(sentenc, dim = 1).cuda()
				sentenc = sentenc.view(batch_size, 1, -1).cuda()
				sentenc = torch.squeeze(sentenc).cuda()
			else:
				sentenc, m = self.encoder(sentences_f[:, t - 1, :, :].permute(1, 0, 2))
				#print(m.size())
				sentenc = torch.unsqueeze(sentenc, dim = 1).cuda()
				sentenc = sentenc.view(batch_size, 1, -1).cuda()
				sentenc = torch.squeeze(sentenc).cuda()

			local_features = local_features.view(batch_size, 1024, -1).cuda()
			sent = self.s(sentenc).cuda()
			vis = self.v(local_features.permute(0 , 2, 1)).cuda()
			sent = torch.unsqueeze(sent, dim = 2).cuda()
			tot = torch.tanh(vis + (sent * self.ones)).cuda()
			a = self.a(tot).cuda()
			a_soft = nn.functional.softmax(a, dim = 2).cuda()
			final = torch.sum(a_soft.permute(0, 2, 1) * local_features, dim = 2).cuda()
			#batch_size = sum([l > t for l in fnd_lengths_sorted])
			for u in range(max(nwf)): # change to max(fnd_words_sorted)
				#batch_size = sum([l > u for l in fnd_word_lengths_sorted])
				#inp = torch.squeeze(fnd[batch_size, t, u, fill_dim])
				h, c = self.LSTM1(final.cuda(), (h, c))
				h1, c1 = self.LSTM2(h, (h1, c1))
				word = self.f(h1)
				word_list[:, t, u, :] = word
			
		return word_list


resnet = Resnet()
sentenceg = SentenceG(vocab_dim, 512, 1024) # vocab_dim here is 3266 which has to be shortened to remove the missing words. 
sentenceencoder = SentenceEncoder(vocab_dim, 1024) # sentencs = sentenc(sentenceg_o[:, 0, :, :].permute(1, 0, 2).cuda()) That is how input is to be given.
model = SentenceAttentionalDecoder(1024, 1024, sentenceencoder, sentenceg, resnet)

for x, y in enumerate(training_set):
	#print(y[4:])
	word_list = model(y).cuda()
	print(word_list.size())
	break

	