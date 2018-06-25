import tensorflow as tf
import numpy as np
import copy
from process_data import process_data as pd
import random

class Model :
	def __init__(self, char_size, word_size, score_size, word_max_len, sentence_max_len, batch_size, char_dict, rnn) :
		#set out_side
		#embedding
		self.char_size = char_size
		self.word_size = word_size
		#get_output
		self.score_size = score_size

		self.word_max_len = word_max_len
		self.sentence_max_len = sentence_max_len

		self.batch_size = batch_size
		
		#set_inside
		#embedding
		self.char_dim = 100 #char_cnn
		self.word_dim = 300
		#full_rnn
		self.lstm_size = 300


		###
		self.char_dict = char_dict
		self.rnn = rnn

		self.out_put_data = []


	def Set_placeholder(self) :
		#embedding
		self.char_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sentence_max_len, self.word_max_len],
			name = 'char_ids')                              ## batch, sentence, word
		self.word_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sentence_max_len], 
			name = 'word_ids')                              ## batch, sentence


		### TODO : it really need? i dont know not yet
		# self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
		# 				name="word_lengths")

		#char_cnn
		self.cnn_dropout = tf.placeholder(tf.float32, name='cnn_dropout')

		#full_rnn
		self.sequence_length = tf.placeholder(tf.int32, shape=[self.batch_size],
			name='sequence_length')							## batch

		self.rnn_dropout = tf.placeholder(tf.float32, name='rnn_dropout')

		#get_output
		self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.score_size],
			name = "labes")						 	## batch, sentence == something wrong...		

	# def Embedding(self) :
		# with tf.variable_scope("char_embedding") :
		# 	self.char_embedding = tf.get_variable("char_embedding", [self.char_size, self.char_dim]) 
		# 	self.char_embedding = tf.nn.embedding_lookup(self.char_embedding, self.char_ids, name='pro_char_embedding')
		# 	print("pre_char_embedding : " ,self.char_embedding)
		# 	### (?, ?, ?, 100)

		# 	temp = tf.shape(self.char_embedding)
		# 	self.char_embedding = tf.reshape(self.char_embedding, 
		# 		shape = [temp[0]*temp[1], temp[-2], self.char_dim])

		# 	self.char_embedding = tf.expand_dims(self.char_embedding, -1)
		# 	print('char_embedding : ',self.char_embedding)

		# 	### (? [, ?, 100)
			
		# 	#word_lengths = tf.reshape(self.word_lengths, shape = [temp[0]*temp[1]])
			
		# self.Char_cnn()

		# with tf.variable_scope("word_embedding") :
		# 	self.word_embedding = tf.get_variable('word_embedding', [self.word_size, self.word_dim])	
		# 	self.word_embedding = tf.nn.embedding_lookup(self.word_embedding, self.word_ids, name= "pro_word_embedding")
		# 	print("word_embedding : " ,self.word_embedding)
		# 	### (?, ?, 300)

		# self.Feature_concat()
		### (?, ?, 300 + cnn_feature)
	
	def Embedding(self) :
		self.full_feature = []

		with tf.variable_scope('embedding') :
			self.char_embedding = tf.get_variable('char_embedding', [self.char_size, self.char_dim])
			self.word_embedding = tf.get_variable('word_embedding', [self.word_size, self.word_dim])

			char_in = tf.split(self.char_ids, self.sentence_max_len, 1)
			#print(self.char_ids)
			## 30 37 28 -> 30 1 28 ....?
			#print('char_in', char_in)
			#word_in = tf.split(tf.expand_dims(self.word_ids, -1), self.sentence_max_len, 1)

			for i in range(self.sentence_max_len) :
				char_i = tf.reshape(char_in[i], [-1, self.word_max_len])
				
				#word_i = tf.reshape(word_in[i], [-1, 1])

				
				self.char_embedd = tf.nn.embedding_lookup(self.char_embedding, char_i)

				#self.word_embedd = tf.nn.embedding_lookup(self.word_embedding, word_i)
				#self.word_feature = tf.concat([self.h_pool, tf.squeeze(self.word_embedd, [1])], axis=1)
				#self.word_feature = tf.expand_dims(self.word_feature, 1) # (30 1 420)
				self.full_feature.append(self.Char_cnn())
			
			self.word_embedd = tf.nn.embedding_lookup(self.word_embedding, self.word_ids)
			self.full_feature = tf.stack(self.full_feature, axis=1)

			self.full_feature = tf.concat([self.word_embedd, self.full_feature], axis=-1)
			print('char_i : ', char_i)
			#print('word_i: '  , word_i)
			print('pool : ', self.pool)
			#print('h_pool : ', self.h_pool)
			print('char_embedd : ', self.char_embedd)
			print('word_embedd : ', self.word_embedd)
			#print('word_feaure : ', self.word_feature)
			print(self.full_feature)




	def Char_cnn(self) :
		self.char_embedd = tf.expand_dims(self.char_embedd, -1)

		layers= []
		filter_sizes = [3, 6, 9]
		feature_map = [20, 40, 60]
		for i, filter_size in enumerate(filter_sizes) :
			reduced_len = self.char_embedd.get_shape()[1] - filter_size + 1

			w = tf.Variable(tf.truncated_normal([filter_size, self.char_dim, 1, feature_map[i]], stddev=0.1), name='w')

			conv = tf.nn.conv2d(
				self.char_embedd,
				w, 
				strides=[1,1,1,1],
				padding= 'VALID'
				)

			self.pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_len, 1, 1], [1,1,1,1], 'VALID')

			layers.append(tf.squeeze(self.pool))

		h_pool = tf.concat(layers, axis=-1)

		return h_pool



	# def Char_cnn(self) :
		
	# 	num_filters = 50
	# 	filter_sizes = [3,6,9]
	# 	cnn_dim = 6
	# 	pooled_output = []
				
	# 	with tf.variable_scope('char_cnn') :

	# 		for i, filter_size in enumerate(filter_sizes) :
	# 			filter_shape = [filter_size, self.char_dim, 1, num_filters]

	# 			w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="char_w")
	# 			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="char_b")

	# 			conv = tf.nn.conv2d(
	# 				self.char_embedding, 
	# 				w,
	# 				strides=[1, 1, 1, 1],
	# 				padding="VALID",
	# 				name="char_conv")

	# 			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="char_relu")
	# 			# Maxpooling over the outputs
	# 			self.pooled = tf.nn.max_pool(
	# 				h,
	# 				ksize = [1, 1,1,1],  ## char_기준 max_word_len max_len - filter_size + 1
	# 				##.... 모르겠는데???
	# 				strides=[1, 1, 1, 1],
	# 				padding='VALID',
	# 				name="char_pool")
			
	# 			pooled_output.append(self.pooled)

	# 		print(pooled_output)
	# 		self.num_filter_total = num_filters * len(filter_sizes)

	# 		self.h_pool = tf.concat(pooled_output, axis=3) ## idont know it is right

	# 		print('h_pool : ', self.h_pool)
	# 		self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filter_total])
	# 		print('h_pool_flat : ', self.h_pool_flat)
	# 		self.char_feature = self.h_pool_flat## TODO : have to change shape
	# 		print("pre_char_feature : " ,self.char_feature)

	# 		temp = tf.shape(self.char_embedding)
			
	# 		print("char_feature : " ,self.char_feature)


	# def Feature_concat(self) :
	# 	temp = tf.shape(self.word_embedding)
	# 	self.char_feature = tf.reshape(self.char_feature, shape = [temp[0], temp[1], self.num_filter_total])

	# 	self.full_feature = tf.concat([self.word_embedding, self.char_feature], axis = -1) # 한줄 펴기?
	# 	print("full_feature : " ,self.full_feature)

	
	def Full_rnn(self) :
		with tf.variable_scope('full_rnn') :
			cell_fw = tf.contrib.rnn.LSTMCell(self.lstm_size)
			cell_bw = tf.contrib.rnn.LSTMCell(self.lstm_size)
			(output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw, cell_bw, self.full_feature, #self.word_char_embeddings,
				sequence_length=self.sequence_length, dtype=tf.float32)

			self.output = tf.concat([output_fw, output_bw], axis=-1)
			self.output = tf.nn.dropout(self.output, self.rnn_dropout)

			print('pre_output : ',self.output)

	# def Full_rnn(self) :
	# 	with tf.variable_scope('full')


	def Full_cnn(self) :
		filter_sizes = [3, 6, 9]
		filter_map = 20
		pooled_output = []
		with tf.variable_scope('full_cnn') :
			self.full_feature = tf.expand_dims(self.full_feature, -1)

			for i , filter_size in enumerate(filter_sizes) :
				filter_shape = [filter_size, self.word_dim + 120, 1, filter_map]

				w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
				b = tf.Variable(tf.constant(0.1, shape = [filter_map]))

				conv = tf.nn.conv2d(
					self.full_feature,
					w,
					strides=[1,1,1,1], 
					padding='VALID',
					)

				h = tf.nn.relu(tf.nn.bias_add(conv, b))
				pooled = tf.nn.max_pool(
					h,
					ksize = [1, self.sentence_max_len-filter_size+1, 1, 1], 
					strides = [1,1,1,1], 
					padding='VALID')

				pooled_output.append(pooled)

			self.num_filters_total = filter_map * len(filter_sizes)
			h_pool_out = tf.concat(pooled_output, axis=3)
			self.h_pool_flat = tf.reshape(h_pool_out, [-1, self.num_filters_total])

			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.rnn_dropout)

	def Cnn_get_output(self) :
		with tf.variable_scope('cnn_out') :
			w = tf.get_variable('full_w', dtype = tf.float32, shape = [self.num_filters_total, self.score_size])
			b = tf.Variable(tf.constant(0.1, shape=[self.score_size]))

			self.l2_loss = tf.constant(0.0)

			self.l2_loss += tf.nn.l2_loss(w)
			self.l2_loss += tf.nn.l2_loss(b)

			self.score = tf.nn.xw_plus_b(self.h_drop, w, b)
			self.pred = tf.argmax(self.score, 1)

			losses = tf.nn.softmax_cross_entropy_with_logits(logits= self.score, labels= self.labels)
			self.loss = tf.reduce_mean(losses)

			tf.summary.scalar('loss', self.loss)

			correct_predictions = tf.equal(self.pred, tf.argmax(self.labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def Rnn_get_output(self) :
		with tf.variable_scope('get_output') :
			w = tf.get_variable('full_w', dtype = tf.float32, shape = [2*self.lstm_size, self.score_size])
			b = tf.get_variable('full_b', dtype = tf.float32, shape = [self.score_size], initializer=tf.zeros_initializer())

			#self.output = tf.reshape(self.output, [ :, -1, :])
			self.output = self.output[:, -1, :]

			print('output : ',self.output)

			self.score = tf.nn.xw_plus_b(self.output, w, b, name='score')


			print('score : ', self.score)
			self.pred = tf.argmax(self.score, 1, name='pred')
			print('pred : ', self.pred)

			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels = self.labels)
			self.loss = tf.reduce_mean(losses)

			tf.summary.scalar('loss', self.loss) 

			correct_predictions = tf.equal(self.pred, tf.argmax(self.labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def Train_op(self) :
		optimizer = tf.train.AdamOptimizer(0.001)
		self.train_op = optimizer.minimize(self.loss)


	def Initial_session(self) :
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()


	def Build(self) :
		self.Set_placeholder()
		self.Embedding()

		if self.rnn : 
			self.Full_rnn()
			self.Rnn_get_output() # set logit and loss
		else :
			self.Full_cnn()
			self.Cnn_get_output()
		
		self.Train_op()
		self.Initial_session()





	def Train(self, data, epoch_num, word_reverse_dict) :
		self.merged = tf.summary.merge_all()

		for epoch in range(epoch_num) :
			print('epoch : ', epoch)
			self.run_epoch(data, word_reverse_dict)

			if epoch % 10 == 0 :
				self.saver.save(self.sess, './save')

		print(self.out_put_data)

	def run_epoch(self, data, word_reverse_dict) :
		batches, valids = self.Get_batch(data)

		for i, batch in enumerate(batches) :
			#print(len(batch[0]))
			fd = self.Get_feed_dict(batch, word_reverse_dict)
			
			_, train_loss, summary, acc, pred, score, temp = self.sess.run([self.train_op, self.loss, self.merged, self.accuracy, self.pred, self.score, self.labels], feed_dict=fd)
			#print(temp)
			#print(pred)
			#print(score)
			if i%50 == 0 :
				print('batch : ' + str(i))
				print(acc, train_loss)
			
				#print(summary)
				#print(_)

		print('start_valid')

		valid_loss = []
		valid_acc = []

		for i, valid in enumerate(valids)  :
			fd = self.Get_feed_dict(valid, word_reverse_dict)

			_, loss, summary, acc = self.sess.run([self.train_op, self.loss, self.merged, self.accuracy], feed_dict=fd)
			

			valid_loss.append(loss)
			valid_acc.append(acc)

		lo = np.mean(np.array(valid_loss))
		va = np.mean(np.array(valid_acc))

		print(lo)
		print(va)

		self.out_put_data.append([lo,va])
			## ...

	def Get_batch(self, datas) :
		#print(datas[0])
		###			data	sentence
		### data = [[label, [[[char_tegs], words], ....]],...]
		random.shuffle(datas) # 최종으로 섞음

		batch_data = []
		valid_data = []
		datas = datas[:5400]
		train_datas = datas[:int(5400 *0.8)]
		valid_datas = datas[int(5400*0.8):]

		data_devie = int(len(train_datas) / self.batch_size)
		
		for i in range(data_devie) :
			words = []
			labels = []
			for j in range(i *self.batch_size, (i+1)*self.batch_size) :
				labels.append(train_datas[j][0])
				words.append(train_datas[j][1])

			if (i+1)*self.batch_size >= len(train_datas)-1 :
				break

			batch_data.append([copy.deepcopy(labels), copy.deepcopy(words)])

		print('total_batch : ', data_devie)

		valid_devdie = int(len(valid_datas) / self.batch_size)

		for i in range(valid_devdie) :
			words = []
			labels = []
			for j in range(i *self.batch_size, (i+1)*self.batch_size) :
				labels.append(valid_datas[j][0])
				words.append(valid_datas[j][1])

			if (i+1)*self.batch_size >= len(valid_datas)-1 :
				break

			valid_data.append([copy.deepcopy(labels), copy.deepcopy(words)])

		#print(valid_data[-1][1])
		
		return batch_data, valid_data
			

	def Get_feed_dict(self, batch_data, word_reverse_dict) :
		#batch_data = [[[[char_tags...], word_tag], ....], []...]
		__ = 0
		word_tag_list = []
		char_tag_list = []

		for sentence in batch_data[1] : #sentence = [[[char_tag], word],...]
			if __ < len(sentence) :
				__ = len(sentence)
			word_tag_one_sentence = []
			char_tag_one_sentence = []

			for word in sentence :    #word = [[char], word]
				char_tag_one_sentence.append(word[0])
				word_tag_one_sentence.append(word[1])

			char_tag_list.append(copy.deepcopy(char_tag_one_sentence))
			word_tag_list.append(copy.deepcopy(word_tag_one_sentence))

		word_lens, _ = pd.Get_word_lens(word_tag_list, word_reverse_dict)
				#word_max_len
		word_tag_list, word_lens, ___ = pd.Sentence_padding(self.sentence_max_len, word_tag_list, word_lens)
		char_tag_list  = pd.Word_padding(self.word_max_len, sentence_max_len, char_tag_list)

		sequence_length = []
		for x in range(self.batch_size) :
			sequence_length.append(sentence_max_len)
		
		#print(word_tag_list)

		#print(np.array(word_tag_list).shape)
		#print(np.array(char_tag_list).shape)
		#print(np.array(word_lens).shape)
		#print(np.array(sentence_lens).shape)

		feed = {
			self.char_ids : char_tag_list,
			#self.word_lengths : word_lens,
			self.word_ids : word_tag_list,
			self.sequence_length : sequence_length
		}

		labels = []
		for i in range(len(batch_data[0])) :
			temp = [0, 0, 0, 0, 0, 0]
			labels.append(copy.deepcopy(temp))

		for i, var in enumerate(batch_data[0]) :
			labels[i][int(var)] = 1


		feed[self.labels] = labels

		feed[self.cnn_dropout] = 0.5
		feed[self.rnn_dropout] = 0.5

		return feed

	def Test(self, data, word_reverse_dict ) :
		loss_list = []
		acc_list = []

		for i, batch in enumerate (self.Get_test_batch(data)) :
			fd = self.Get_feed_dict(batch, word_reverse_dict)
			_, loss, acc = self.sess.run([self.train_op, self.loss, self.accuracy], feed_dict = fd)

			loss_list.append(loss)
			acc_list.append(acc)

		print(np.mean(np.array(loss_list)))
		print(np.mean(np.array(acc_list)))

	def Get_test_batch(self, datas) :
		#print(datas)

		random.shuffle(datas, random.random) # 최종으로 섞음

		batch_data = []
	
		data_devie = int(len(datas) / self.batch_size)
		
		for i in range(data_devie) :
			words = []
			labels = []
			for j in range(i *self.batch_size, (i+1)*self.batch_size) :
				labels.append(datas[j][0])
				words.append(datas[j][1])

			if (i+1)*self.batch_size > len(datas) :
				break

			batch_data.append([copy.deepcopy(labels), copy.deepcopy(words)])


		return batch_data


if __name__ == "__main__" :
	pd = pd()

	rnn = False

	data_path = pd.Get_data_path('TREC')

	train_datas = pd.Get_data(data_path[0])
	test_datas = pd.Get_data(data_path[2])
	
	datas = train_datas + test_datas

	temp = pd.get_param(datas)

	word_list = temp[1]
	word_dict = temp[2]
	word_reverse_dict = temp[3]
	word_size = temp[4]
	char_list = temp[5]
	char_dict = temp[6]
	char_size = temp[7]
	score_list = temp[8]
	score_size = temp[9]
	word_max_len = temp[10]
	sentence_max_len = temp[11]

	print(word_max_len)
	print(sentence_max_len)

	train_data_list = pd.get_param(train_datas)[0]

	data_form = pd.Form_porcessing(train_data_list, word_dict, char_dict)

	print('train_data_size : ',len(data_form))

	md = Model(char_size, word_size, score_size, word_max_len, sentence_max_len, 50, char_dict, rnn)
	md.Build()

	md.Train(data_form, 100, word_reverse_dict)

	test_data_list = pd.get_param(test_datas)[0]

	data_form = pd.Form_porcessing(test_data_list, word_dict, char_dict)

	print('test_data_size : ', len(data_form))

	md.Test(data_form, word_reverse_dict)

