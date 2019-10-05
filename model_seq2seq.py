import tensorflow as tf

class Seq2seq(object):
	
	def build_inputs(self, config):
		self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
		self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
		self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
		self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')

		
	def build_loss(self, logits):
		
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.seq_targets,
			logits=logits,
		)
		loss = tf.reduce_mean(loss)
		return loss

		
	def build_optim(self, loss, lr):
		return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
	
	def attn(self, hidden, encoder_outputs):
		# hidden: B * D
		# encoder_outputs: B * S * D
		attn_weights = tf.matmul(encoder_outputs, tf.expand_dims(hidden, 2))
		# attn_weights: B * S * 1
		attn_weights = tf.nn.softmax(attn_weights, axis=1)
		context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs, [0,2,1]), attn_weights))
		# context: B * D
		return context
				
	def __init__(self, config, w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1):
	
		self.build_inputs(config)
		
		with tf.variable_scope("encoder"):
		
			encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]), dtype=tf.float32, name='encoder_embedding')
			encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
			
			((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=tf.nn.rnn_cell.GRUCell(config.hidden_dim), 
				cell_bw=tf.nn.rnn_cell.GRUCell(config.hidden_dim), 
				inputs=encoder_inputs_embedded, 
				sequence_length=self.seq_inputs_length, 
				dtype=tf.float32, 
				time_major=False
			)
			encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state)
			encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs)
		
		with tf.variable_scope("decoder"):
			
			decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]), dtype=tf.float32, name='decoder_embedding')
					
			with tf.variable_scope("gru_cell"):
				decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
				decoder_initial_state = encoder_state
			
			# if useTeacherForcing and not useAttention:
				# decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
				# decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
				# decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded, initial_state=decoder_initial_state, sequence_length=self.seq_targets_length, dtype=tf.float32, time_major=False)
			
			tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_GO') * w2i_target["_GO"]
			tokens_eos = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_EOS') * w2i_target["_EOS"]
			tokens_eos_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_eos)
			tokens_go_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_go)
			
			W = tf.Variable(tf.random_uniform([config.hidden_dim, config.target_vocab_size]), dtype=tf.float32, name='decoder_out_W')
			b = tf.Variable(tf.zeros([config.target_vocab_size]), dtype=tf.float32, name="decoder_out_b")
			
			def loop_fn(time, previous_output, previous_state, previous_loop_state):
				if previous_state is None:    # time step == 0
					initial_elements_finished = (0 >= self.seq_targets_length)  # all False at the initial step
					initial_state = decoder_initial_state # last time steps cell state
					initial_input = tokens_go_embedded # last time steps cell state
					if useAttention:
						initial_input = tf.concat([initial_input, self.attn(initial_state, encoder_outputs)], 1)
					initial_output = None #none
					initial_loop_state = None  # we don't need to pass any additional information
					return (initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state)
				else:
					def get_next_input():
						if useTeacherForcing:
							prediction = self.seq_targets[:,time-1]
						else:
							output_logits = tf.add(tf.matmul(previous_output, W), b)
							prediction = tf.argmax(output_logits, axis=1)
						next_input = tf.nn.embedding_lookup(decoder_embedding, prediction)
						return next_input
					
					elements_finished = (time >= self.seq_targets_length) 
					finished = tf.reduce_all(elements_finished) #Computes the "logical and" 
					input = tf.cond(finished, lambda: tokens_eos_embedded, get_next_input)
					if useAttention:
						input = tf.concat([input, self.attn(previous_state, encoder_outputs)], 1)
					state = previous_state
					output = previous_output
					loop_state = None

					return (elements_finished, input, state, output, loop_state)
				
			decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
			decoder_outputs = decoder_outputs_ta.stack()
			decoder_outputs = tf.transpose(decoder_outputs, perm=[1,0,2]) # S*B*D -> B*S*D
		
			decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
			decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim))
			decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
			decoder_logits = tf.reshape(decoder_logits_flat, (decoder_batch_size, decoder_max_steps, config.target_vocab_size))
			
		self.out = tf.argmax(decoder_logits, 2)
		
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.seq_targets,
			logits=decoder_logits,
		)
		sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
		loss = loss * sequence_mask
		self.loss = tf.reduce_mean(loss)
		
		self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
			
			
			
			
