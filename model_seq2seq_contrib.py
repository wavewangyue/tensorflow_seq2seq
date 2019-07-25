import tensorflow as tf

class Seq2seq(object):
	
	def build_inputs(self, config):
		self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
		self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
		self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
		self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')
	
	
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
			
			tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_GO') * w2i_target["_GO"]
			
			if useTeacherForcing:
				decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
				helper = tf.contrib.seq2seq.TrainingHelper(tf.nn.embedding_lookup(decoder_embedding, decoder_inputs), self.seq_targets_length)
			else:
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, tokens_go, w2i_target["_EOS"])
				
			with tf.variable_scope("gru_cell"):
				decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
				
				if useAttention:
					if useBeamSearch > 1:
						tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=useBeamSearch)
						tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.seq_inputs_length, multiplier=useBeamSearch)
						attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, memory=tiled_encoder_outputs, memory_sequence_length=tiled_sequence_length)
						decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
						tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
						tiled_decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size*useBeamSearch, dtype=tf.float32)
						tiled_decoder_initial_state = tiled_decoder_initial_state.clone(cell_state=tiled_encoder_final_state)
						decoder_initial_state = tiled_decoder_initial_state
					else:
						attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
						# attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=config.hidden_dim, memory=encoder_outputs, memory_sequence_length=self.seq_inputs_length)
						decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
						decoder_initial_state = decoder_cell.zero_state(batch_size=config.batch_size, dtype=tf.float32)
						decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
				else:
					if useBeamSearch > 1:
						decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=useBeamSearch)
					else:
						decoder_initial_state = encoder_state
			
			if useBeamSearch > 1:
				decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, decoder_embedding, tokens_go, w2i_target["_EOS"],  decoder_initial_state, beam_width=useBeamSearch, output_layer=tf.layers.Dense(config.target_vocab_size))
			else:
				decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=tf.layers.Dense(config.target_vocab_size))
			
			decoder_outputs, decoder_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.seq_targets_length))
			
		if useBeamSearch > 1:
			self.out = decoder_outputs.predicted_ids[:,:,0]
		else:	
			decoder_logits = decoder_outputs.rnn_output
			self.out = tf.argmax(decoder_logits, 2)
			
			sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
			self.loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits, targets=self.seq_targets, weights=sequence_mask)
			
			self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
			
			
