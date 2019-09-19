# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

tf.app.flags.DEFINE_float("lr", 1.0, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 256,
														"Batch size to use during training.")
tf.app.flags.DEFINE_integer("init", 1, "Initialization method")
tf.app.flags.DEFINE_float("dropout_prob", 1.0, "Dropout Probability")
tf.app.flags.DEFINE_integer("decode_method", 0, "0 for Greedy, 1 for Beam")
tf.app.flags.DEFINE_integer("beam_width", 1, "Beam Width incase Beam Search is being used")
tf.app.flags.DEFINE_string("save_dir", "/home/ec2-user/assn/finalcode/data", "Data directory")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,	"Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,	"Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

#variables stroing the flag values
global lr
global learning_rate_decay_factor
global max_gradient_norm
global batch_size
global init
global dropout_prob
global decode_method
global beam_width
global size
global num_layers
global from_vocab_size
global to_vocab_size
global save_dir
global train_dir
global from_train_data
global to_train_data
global from_dev_data
global to_dev_data
global from_test_data
global to_test_data
global max_train_data_size
global steps_per_checkpoint
global decode
global self_test
global use_fp16
lr = FLAGS.lr
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
batch_size = FLAGS.batch_size
init = FLAGS.init
dropout_prob = FLAGS.dropout_prob
decode_method = FLAGS.decode_method
beam_width = FLAGS.beam_width
size = 256
num_layers = 1
from_vocab_size = 44
to_vocab_size= 84
save_dir= FLAGS.save_dir
train_dir= "/home/ec2-user/assn/finalcode/checkpoints/"
from_train_data= None
to_train_data= None
from_dev_data= None
to_dev_data= None
from_test_data= None
to_test_data= None
max_train_data_size= 0
steps_per_checkpoint= int(2*(13125/batch_size))
decode= False
self_test= False
use_fp16= FLAGS.use_fp16




# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (65,80)]


def read_data(source_path, target_path, max_size=None):
	"""Read data from source and target files and put into buckets.

	Args:
		source_path: path to the files with token-ids for the source language.
		target_path: path to the file with token-ids for the target language;
			it must be aligned with the source file: n-th line contains the desired
			output for n-th line from the source_path.
		max_size: maximum number of lines to read, all other will be ignored;
			if 0 or None, data files will be read completely (no limit).

	Returns:
		data_set: a list of length len(_buckets); data_set[n] contains a list of
			(source, target) pairs read from the provided data files that fit
			into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
			len(target) < _buckets[n][1]; source and target are lists of token-ids.
	"""
	data_set = [[] for _ in _buckets]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		with tf.gfile.GFile(target_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				if counter % 100 == 0:
					print("  reading data line %d" % counter)
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()
	return data_set


def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	dtype = tf.float16 if use_fp16 else tf.float32
	model = seq2seq_model.Seq2SeqModel(
			from_vocab_size,
			to_vocab_size,
			_buckets,
			size,
			num_layers,
			max_gradient_norm,
			batch_size,
			lr,
			learning_rate_decay_factor,
			forward_only=forward_only,
			dtype=dtype,
			dropout_prob = dropout_prob)
	ckpt = tf.train.get_checkpoint_state(train_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	"""Train a en->fr translation model using WMT data."""
	early_stop = False
	print("IN TRAIN")
	train_error = open("train_error.log",'w')
	valid_error = open("valid_error.log",'w')
	print(train_dir)
	from_train = None
	to_train = None
	from_dev = None
	to_dev = None
	to_test = None
	from_test = None
	if globals()['from_train_data'] and globals()['to_train_data']:
		from_train_data = globals()['from_train_data']
		to_train_data = globals()['to_train_data']
		from_dev_data = from_train_data
		to_dev_data = to_train_data
		if globals()['from_dev_data'] and globals()['to_dev_data']:
			from_dev_data = globals()['from_dev_data']
			to_dev_data = globals()['to_dev_data']
		from_train, to_train, from_dev, to_dev, from_test, to_test, _, _ = data_utils.prepare_data(
				save_dir,
				from_train_data,
				to_train_data,
				from_dev_data,
				to_dev_data,
				from_test_data,
				to_test_data,
				from_vocab_size,
				to_vocab_size)
	else:
			# Prepare WMT data.
			print("Preparing WMT data in %s" % save_dir)
			from_train, to_train, from_dev, to_dev, from_test, to_test, _, _ = data_utils.prepare_wmt_data(
					 save_dir,  from_vocab_size,  to_vocab_size)

	with tf.Session() as sess:
		# Create model.
		print("Creating %d layers of %d units." % ( num_layers,  size))
		model = create_model(sess, False)

		# Read data into buckets and compute their sizes.
		print ("Reading development and training data (limit: %d)."
					 %  max_train_data_size)
		dev_set = read_data(from_dev, to_dev)
		train_set = read_data(from_train, to_train,  max_train_data_size)
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
		train_total_size = float(sum(train_bucket_sizes))

		# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
		# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
		# the size if i-th training bucket, as used later.
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
													 for i in xrange(len(train_bucket_sizes))]

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		eval_losses_aggr = 0.0
		current_step = 0
		previous_losses = []
		count_early_stop = 0
		best_valid_loss = sys.maxsize

		while True:
			# Choose a bucket according to data distribution. We pick a random number
			# in [0, 1] and use the corresponding interval in train_buckets_scale.
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
											 if train_buckets_scale[i] > random_number_01])

			# Get a batch and make a step.
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
					train_set, bucket_id)
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
																	 target_weights, bucket_id, False)

			step_time += (time.time() - start_time) /  steps_per_checkpoint
			loss += step_loss /  steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step %  steps_per_checkpoint == 0:
				# Print statistics for the previous epoch.
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
				print ("global step %d learning rate %.4f step-time %.2f loss  %.2f perplexity "
							 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
												 step_time, loss, perplexity))
				train_error.write(str("global step %d learning rate %.4f step-time %.2f loss  %.2f perplexity "
							 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
												 step_time, loss, perplexity))+"\n")
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join( train_dir, "translate.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				model_global_step = model.global_step.eval()
				step_time, loss = 0.0, 0.0
				eval_ppx = 0.0
				eval_losses_aggr = 0.0
				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(_buckets)):
					if len(dev_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
						continue
					encoder_inputs, decoder_inputs, target_weights = model.get_batch(
							dev_set, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
																			 target_weights, bucket_id, True)
					eval_losses_aggr += eval_loss
					eval_ppx += math.exp(float(eval_loss)) if eval_loss < 300 else float(
							"inf")
				eval_losses_aggr = eval_losses_aggr/3
				if(early_stop):
					if best_valid_loss < eval_losses_aggr and count_early_stop == 5:
						eval_ppx = eval_ppx/3
						valid_error.write(str("global step %d learning rate %.4f step-time %.2f loss  %.2f perplexity "
									 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
														 step_time, eval_losses_aggr, eval_ppx))+"\n")
						sys.exit()
					elif(best_valid_loss < eval_losses_aggr):
						count_early_stop += 1
					else:
						count_early_stop = 0
						best_valid_loss = eval_losses_aggr
					eval_ppx = eval_ppx/3
					valid_error.write(str("global step %d learning rate %.4f step-time %.2f loss  %.2f perplexity "
								 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
													 step_time, eval_losses_aggr, eval_ppx))+"\n")
				else:
					eval_ppx = eval_ppx/3
					valid_error.write(str("global step %d learning rate %.4f step-time %.2f loss  %.2f perplexity "
								 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
													 step_time, eval_losses_aggr, eval_ppx))+"\n")



				#print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				sys.stdout.flush()

def call_decode_multiple():
	number_of_matches=0
	f_En=open("data/test.En", "r")
	f_Hi=open("data/test.Hi", "r")
	f_En_Lines = f_En.readlines()
	f_Hi_Lines = f_Hi.readlines()
	test_data = []
	truth_data = []
	for line in f_En_Lines:
		test_data.append(line)
	for line in f_Hi_Lines:
		truth_data.append(line)
	test_Hi_Lines = decode_multiple(test_data)
	for index in range(0,len(test_data)):
		print("||"+test_Hi_Lines[index] + "||" + truth_data[index] + "||")
		if test_Hi_Lines[index].strip() == truth_data[index].strip():
			number_of_matches+=1
	accuracy_of_data = float(number_of_matches/(len(test_data)))
	print(accuracy_of_data)

def decode_multiple(set_of_sentences):
	with tf.Session() as sess:
		set_of_outputs = []
		# Create model and load parameters.
		model = create_model(sess, True)
		model.batch_size = 1  # We decode one sentence at a time.

		# Load vocabularies.
		en_vocab_path = os.path.join( save_dir,
																 "vocab%d.from" %  from_vocab_size)
		fr_vocab_path = os.path.join( save_dir,
																 "vocab%d.to" %  to_vocab_size)
		en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
		_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

		# Decode from standard input.
		for sentence in set_of_sentences:
			# Get token-ids for the input sentence.
			token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
			# Which bucket does it belong to?
			bucket_id = len(_buckets) - 1
			for i, bucket in enumerate(_buckets):
				if bucket[0] >= len(token_ids):
					bucket_id = i
					break
			else:
				logging.warning("Sentence truncated: %s", sentence)

			# Get a 1-element batch to feed the sentence to the model.
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(
					{bucket_id: [(token_ids, [])]}, bucket_id)
			# Get output logits for the sentence.
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
																			 target_weights, bucket_id, True)
			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

			# If there is an EOS symbol in outputs, cut them at that point.
			if data_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(data_utils.EOS_ID)]
			# Print out French sentence corresponding to outputs.
			set_of_outputs.append(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
			#print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
			#print("> ", end="")
			sys.stdout.flush()
		return set_of_outputs

def main(_):
	train()

if __name__ == "__main__":
	tf.app.run()
