import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pandas as pd
import glob
import os
import time
import math
import re
import MeCab
import pickle

use_gpu = torch.cuda.is_available()

#コーパスに出現する単語の辞書作成
class Dictionary(object):
	def __init__(self):
		self.word2idx={}
		self.idx2word=[]
		self.wordscount={}
		self.unknown_word_threshold = 3
		self.unknown_word_symbol = "<unk>"
		self.k=['<bos>','<eos>',self.unknown_word_symbol]
		for i in self.k:
			self.idx2word.append(i)
			self.word2idx[i]=len(self.idx2word)-1
	def add_word(self,word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word]=len(self.idx2word)-1
			self.wordscount[word] = self.wordscount.get(word,0)+1
		else:
			self.wordscount[word] = self.wordscount.get(word,0)+1
	
	def refresh(self):
		word_id = 0
		dictionary = dict()
		for i in self.k:
			dictionary[i] = word_id
			word_id += 1
		for word,count in sorted(self.wordscount.items(), reverse=True):
			if count <= self.unknown_word_threshold:
				continue
			dictionary[word] = word_id
			word_id += 1
		dictionary[self.unknown_word_symbol] = word_id
		self.word2idx = dictionary

	def __len__(self):
		return len(self.idx2word)

#読み込み
class Corpus(object):
	def __init__(self):
		#辞書および各種データを構築済みの場合は対象データをそのまま読み込み
		pkl_file='data.pkl'
		if os.path.exists(pkl_file): 
			with open(pkl_file, 'rb') as f:
				data=pickle.load(f)
				self.Contents_Train=data['Contents_Train']
				self.Contents_Valid=data['Contents_Valid']
				self.Contents_Test=data['Contents_Test']
				self.dictionary=data['Vocablary']
				print("Dictionary & Data is successfully loaded")
		else:
			self.dictionary = Dictionary()
			self.unknown_word_symbol = Dictionary().unknown_word_symbol
			self.corpus_files = 'Train_Data.html'
			self.valid_files = 'Valid_Data.html'
			self.test_files = 'Test_Data.html'
			Contents_Train=[]
			Contents_Valid=[]
			Contents_Test=[]
			Contents ={self.corpus_files:Contents_Train,self.valid_files:Contents_Valid,self.test_files:Contents_Test}
			#前処理（特殊文字などの除去、<bos><eos>の付与）
			jisx0208 = []
			with open("unicode.txt", "r", encoding="utf-8_sig") as f:
				for line in f:
					line = line.strip()
					jisx0208.append(line)
			jisx0208.append('\n')
			jisx0208.append('　')
			jisx0208.append(' ')
			jisx0208.append(' ')
			jisx0208.append('‎')
			jisx0208.append(' ‎')
			Jisx0208 = set(jisx0208)
			for files in Contents.keys():
				with open(files,'r') as f:
					for line in f:
						if line not in Jisx0208 and re.search(r'=[^=]+=',line) is None:
							line=line.strip()
							line=re.sub('[0-9０-９]+','*',line)
							line=re.sub('[=]+[^=][\S]+[=]+','',line)
							line=re.sub('（[^）]+）','',line)					
							line=re.sub('[A-Zａ-ｚ]+','',line)
							a=''
							for g in line:
								a +=g
								if g == '。' and a != '。' and a not in Contents[files]:
									Contents[files].append(a)
									a=''
								elif a == '。':
									a=''
				n_gram1=[]
				for sen in Contents[files]:
					sen = re.sub(r"\s+", "", sen)	
					tokens = MeCab.Tagger('-Owakati').parse(sen).strip().split()
					n=1
					for i in range(len(tokens)-n+1):
						if i==0:
							n_gram1.append("<bos>")
						n_gram1.append(tokens[i:i+n])
					n_gram1.append("<eos>") 
				Contents[files]=n_gram1
			self.Contents_Train=Contents[self.corpus_files]
			self.Contents_Valid=Contents[self.valid_files]
			self.Contents_Test=Contents[self.test_files]
			
			self.Contents_Train=self.tokenize1(self.Contents_Train)
			#self.Contents_Valid=self.tokenize2(self.Contents_Valid)
			#self.Contents_Test=self.tokenize2(self.Contents_Test)
			
			self.Contents_Train=self.assign_ids(self.Contents_Train)
			self.Contents_Valid=self.assign_ids(self.Contents_Valid)
			self.Contents_Test=self.assign_ids(self.Contents_Test)

			#データを保存
			data={}
			data['Contents_Train']=self.Contents_Train
			data['Contents_Valid']=self.Contents_Valid
			data['Contents_Test']=self.Contents_Test
			data['Vocablary']=self.dictionary
			with open(pkl_file,'wb') as f:
				pickle.dump(data,f,-1)
				print("Dictionary is save to dataF.pkl")
	#各単語を辞書に登録
	def tokenize1(self,texts):
		print(texts)
		for text in texts:	
			if text == '<eos>' or text == '<bos>':
				word = text
			else:
				word = text[0]
				self.dictionary.add_word(word)

		#self.dictionary.refresh()

		return texts
	#不要な関数
	def tokenize2(self,texts):
		print(texts)
		for text in texts:	
			if text == '<eos>' or text == '<bos>':
				word = text
			else:
				word = text[0]

		return texts
	#データ中の各文字をidへ変換
	def assign_ids(self,texts):
		tokens = len(texts)
		ids = torch.LongTensor(tokens)
		token = 0		
		for text in texts:
			if text == '<eos>' or text == '<bos>':
				word = text
			else:
				word = text[0]
			if word in self.dictionary.word2idx:
				ids[token] = self.dictionary.word2idx[word]
			else:
				ids[token] = self.dictionary.word2idx[self.unknown_word_symbol]
			token +=1

		return ids


#RNNモデルの定義
class RNNModel(nn.Module):
	
	def  __init__(self,rnn_type, ntoken, ninp, nhid, nlayers, dropout = 0.5, tie_weights=False):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn=getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH':'tanh','RNN_RELU': 'relu'}[rnn_type]
			except keyError:
				raise ValueError("""An invalid option for '--model' was supplied, options are ['LSTM','GRU','RNN_TANH' or 'RNN_RELU']""" )
			self.rnn = nn.RNN(ninp,nhid,nlayers, nonlinearity=nonlinearity, dropout=dropout)
		self.decoder = nn.Linear(nhid, ntoken)

		# Optionally tie weights as in:\n",
		# \"Using the Output Embedding to Improve Language Models\" (Press & Wolf 2016)\n",
		# https://arxiv.org/abs/1608.05859\n",
		# and\n",
		# \"Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling\" (Inan et al. 2016)\n",
		# https://arxiv.org/abs/1611.01462\n",

		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight

		self.init_weights()

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange=0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.fill_(0)
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden):
		emb = self.drop(self.encoder(input))
		output, hidden = self.rnn(emb, hidden)
		output = self.drop(output)
		decoded = self.decoder(output.view(output.size(0)*output.size(1),output.size(2)))
		return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type=='LSTM':
			return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
		else:
			return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

#コーパス読み込み
corpus = Corpus()

def batchify(data, bsz):
	# Work out how cleanly we can divide the dataset into bsz parts.
	nbatch = data.size(0) // bsz
	# Trim out any extra elements that wouldn't cleanly fit (remainders).
	data = data.narrow(0,0, nbatch*bsz)
	# Evenly divide the data across the bsz batches.
	data = data.view(bsz, -1).t().contiguous()
	if use_gpu:
		data = data.cuda()
	return data

# 各データをバッチに分割
batch_size = 20
eval_batch_size = 20 #10
train_data = batchify(corpus.Contents_Train, batch_size)
val_data = batchify(corpus.Contents_Valid, eval_batch_size)
test_data = batchify(corpus.Contents_Test, eval_batch_size)

#モデルの設定
emsize = 200
nhid = 200
nlayers = 2
dropout = 0.3
tied = False
ntokens = len(corpus.dictionary)
model = RNNModel('LSTM', ntokens, emsize, nhid, nlayers, dropout, tied)
if use_gpu:
	model.cuda()

criterion = nn.CrossEntropyLoss()

#訓練
bptt = 35 #最大35までのシーケンスを利用
lr = 10 
clip = 0.25
log_interval = 200

def repackage_hidden(h):
	"""Wraps hidden states in new Variables, to detach them from their history."""
	if isinstance(h, Variable):#修正箇所
		return Variable(h.data)
	else:
		return tuple(repackage_hidden(v) for v in h)

# get_batch subdivides the source data into chunks of length args.bptt.\n",
# If source is equal to the example output of the batchify function, with\n",
# a bptt-limit of 2, we'd get the following two Variables for i = 0:\n",
# ┌ a g m s ┐ ┌ b h n t ┐\n",
# └ b h n t ┘ └ c i o u ┘\n",
# Note that despite the name of the function, the subdivison of data is not\n",
# done along the batch dimension (i.e. dimension 1), since that was handled\n",
# by the batchify function. The chunks are along dimension 0, corresponding\n",
# to the seq_len dimension in the LSTM.\n",	

def get_batch(source, i, evaluation=False):
	seq_len=min(bptt, len(source) -1 -i)
	if evaluation:
		with torch.no_grad():
			data=Variable(source[i:i+seq_len])#修正箇所
	else:
		data=Variable(source[i:i+seq_len])
	target = Variable(source[i+1:i+1+seq_len].view(-1))
	return data, target

def evaluate(data_source):
	#Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0
	ntokens = len(corpus.dictionary)
	hidden = model.init_hidden(eval_batch_size)
	for i in range(0, data_source.size(0)-1, bptt):
		data, targets = get_batch(data_source, i, evaluation=True)
		output, hidden = model(data, hidden)
		output_flat = output.view(-1, ntokens)
		total_loss += len(data)*criterion(output_flat, targets).data
		hidden = repackage_hidden(hidden)
	return total_loss.item()/len(data_source)#修正箇所

def train():
	#Turn on training mode which enables dropout.
	model.train()
	total_loss=0
	start_time = time.time()
	ntokens=len(corpus.dictionary)
	hidden = model.init_hidden(batch_size)
	for batch,i in enumerate(range(0, train_data.size(0)-1, bptt)):
		data,targets = get_batch(train_data, i)
		#Starting each batch, we detach the hidden state from how it was previously produced.
		#If we didn't, the model world try backprogating all the way to start of the dataset.
		hidden=repackage_hidden(hidden)
		model.zero_grad()
		output, hidden = model(data, hidden)
		loss = criterion(output.view(-1,ntokens), targets)
		loss.backward()

		#'clip_grad_norm' helps prevent the exploding gradient problem in RNNs/LSTMs.
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		for p in model.parameters():
			p.data.add_(-lr, p.grad.data)

		total_loss += loss.data
		
		if batch % log_interval == 0 and batch > 0:
			cur_loss = total_loss.item() / log_interval#修正箇所
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} |''loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // bptt, lr, elapsed*1000 /log_interval, cur_loss, math.exp(cur_loss)))
			total_loss = 0
			start_time=time.time()

#Loop over epochs.
lr = lr
best_val_loss = None

# early stopping
#num_halved = 0
num_worse = 0
early_stopping=3

#At any point you can hit Ctrl + C to break out of training early.
epochs = 30
try:
	for epoch in range(1, epochs+1):
		epoch_start_time = time.time()
		train()
		val_loss = evaluate(val_data)
		print('-'*89)
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ''valid ppl {:8.2f}'.format(epoch, (time.time()-epoch_start_time),val_loss, math.exp(val_loss)))
		print('-'*89)
		#Save the model if the validation loss is the best we've seen so far.
		if not best_val_loss or val_loss < best_val_loss:
			with open('model.pt', 'wb') as f:
				torch.save(model, f)
			best_val_loss = val_loss
			num_worse = 0
		else:
			#Anneal the learning rate if no improvement has been senn in the validation dataset.
			lr /=4.0
			num_worse += 1
			#num_halved += 1
    		# early stopping
		if num_worse >= early_stopping:
			print("\n\nPerplexity (valid) failed to improve for {} epochs.".format(early_stopping))
			print("Stopped training.\n\n")
			break

except KeyboardInterrupt:
	print('-'*89)
	print('Exiting from training early')

#Load the best saved model.
with open('model.pt','rb') as f:
	model = torch.load(f)

#Run on test data.
test_loss = evaluate(test_data)
print('='*89)
print('|End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('='*89)

#ランダムな文章を生成
temperature=1.0
for j in range(10):
	#隠れ層はランダムに初期化
	hidden = model.init_hidden(1)

	#<bos>から始める
	with torch.no_grad():
		input = Variable(torch.LongTensor([[corpus.dictionary.word2idx['<bos>']]])).cuda()
	model = model
	results = []
	for i in range(100):
		output, hidden = model(input, hidden)
		word_weights = output.squeeze().data.div(temperature).exp()
		word_idx=torch.multinomial(word_weights,1)[0]
		input.data.fill_(word_idx)
		word = corpus.dictionary.idx2word[word_idx]

		#<eos>が出たら文末の合図
		if word == '<eos>':
			break

		results.append(word)
	print(' '.join(results))









