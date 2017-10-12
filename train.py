import os
import sys
import numpy as np
import tensorflow as tf
import time
import data_utils


class FLAGS():
  pass


FLAGS.batch_size = 20
FLAGS.hops = 2
FLAGS.value_reading = '(q + o).R'
FLAGS.mem_size = 1000
FLAGS.embedding_size = 500
FLAGS.initial_learning_rate = 0.05
FLAGS.max_gradient_norm = 40.0
FLAGS.num_neg_samples = 20000
FLAGS.l2_reg_strength = 0.00001
FLAGS.keep_prob = 1.0
FLAGS.loss = 'sampled_softmax'   # 'sampled_softmax' or 'margin' or 'nce_loss'
                                 # or 'NLL'
FLAGS.optimizer = 'adagrad'
FLAGS.result_embeddings_separate = False
FLAGS.normalize_embeddings = False
FLAGS.length_normalization = True

FLAGS.docs_count = 0
FLAGS.pad_word = u'_NIL'
FLAGS.max_ans_count = 4

FLAGS.mode = 'train'   # 'train' or 'test_debug' or 'test' or 'train_1_q'
FLAGS.data_pkl_file = 'processed_data_{}.pkl'.format(FLAGS.docs_count)

FLAGS.data_dir = '/home/qv/nlp-data/kv_memn2n_data/data'

# FLAGS.train_dir_base = '/Users/qv/nlp-data/kv_memn2n_data/train'
'''
FLAGS.train_dir_base = '/home/qv/nlp-data/kv_memn2n_data/train'
FLAGS.train_dir = "{}-d={}-l={}-m={}-h={}-v={}-o={}".format(
  FLAGS.train_dir_base, FLAGS.docs_count, FLAGS.initial_learning_rate,
  FLAGS.mem_size, FLAGS.hops, FLAGS.value_reading, FLAGS.optimizer)
'''
FLAGS.train_dir = '/home/qv/nlp-data/kv_memn2n_data/train'

FLAGS.input_dir = '/home/qv/nlp-data/kv_memn2n_data/movieqa'
FLAGS.entities_file = 'knowledge_source/entities.txt'
FLAGS.win_file = 'wiki-w=0-d=3-i-m.txt'


def add_gradient_noise(t, stddev=1e-3, name=None):
    gn = tf.random_normal(tf.shape(t), stddev=stddev)
    return tf.add(t, gn, name=name)


class KVMemN2N_Model(object):
  def __init__(self, forward_only, data_dict):
    self.data_dict = data_dict
    vocab = data_dict['vocab']
    vocab_size = len(vocab)
    if FLAGS.result_embeddings_separate:
      ans_vocab = data_dict['ans_vocab']
      ans_vocab_size = len(ans_vocab)
    else:
      ans_vocab = vocab
      ans_vocab_size = vocab_size
    max_q_len = data_dict['max_q_len']
    max_sentence_len = data_dict['max_sentence_len']
    # unigram_freq = data_dict['unigram_freq']

    # Debugging
    self.debug_dict = {}

    # Placeholders for input
    self.q_b = tf.placeholder(tf.int32,
                              [None, max_q_len],
                              name='questions')
    self.q_len_b = tf.placeholder(tf.float32,
                              [None, 1],
                              name='questions_len')
    self.ans_b = tf.placeholder(tf.int64,
                                [None, FLAGS.max_ans_count],
                                name='answers')
    if FLAGS.loss == 'NLL':
      self.neg_samples = tf.placeholder(tf.int32,
                                        [None, FLAGS.num_neg_samples],
                                        name='negative_answer_samples')
      
    self.mem_key_b = tf.placeholder(tf.int32,
                                    [None, FLAGS.mem_size, max_sentence_len],
                                    name='mem_key')
    self.key_len_b = tf.placeholder(tf.float32,
                                    [None, FLAGS.mem_size, 1],
                                    name='key_len')
    self.mem_val_b = tf.placeholder(tf.int32,
                                    [None, FLAGS.mem_size],
                                    name='mem_key')
    self.mem_wts_b = tf.placeholder(tf.float32,
                                    [None, FLAGS.mem_size],
                                    name='mem_wts')

    # Embedding Matrix
    self.A_var = tf.get_variable(
      'A', shape=[vocab_size - 1, FLAGS.embedding_size],
      initializer=tf.contrib.layers.xavier_initializer())
    nil_word_slot = tf.zeros([1, FLAGS.embedding_size])
    self.A = tf.concat(0, [nil_word_slot, self.A_var])

    # [batch_sz, max_q_len, embedding_sz]
    query_embedding = tf.nn.embedding_lookup(self.A, self.q_b)
    # [batch_sz, mem_sz, max_sentence_len, embedding_sz]
    key_embedding = tf.nn.embedding_lookup(self.A, self.mem_key_b)
    # [batch_sz, mem_sz, 1, embedding_sz]
    val_embedding = tf.nn.embedding_lookup(self.A, self.mem_val_b)

    # Dropout
    if not forward_only and FLAGS.keep_prob < 1.0:
      query_embedding = tf.nn.dropout(query_embedding,
                                      keep_prob=FLAGS.keep_prob)
      key_embedding = tf.nn.dropout(key_embedding,
                                    keep_prob=FLAGS.keep_prob)
      val_embedding = tf.nn.dropout(val_embedding,
                                    keep_prob=FLAGS.keep_prob)

    # Bag of words model - sum embeddings across words in a question/key/value
    # [batch_sz, embedding_sz]
    if FLAGS.length_normalization:
      q_len = tf.expand_dims(self.q_len_b, -1)
      q = query_embedding / q_len
    q = tf.reduce_sum(query_embedding, 1)
    
    # [batch_sz, mem_sz, embedding_sz]
    if FLAGS.length_normalization:
      k_len = tf.expand_dims(self.key_len_b, -1)
      k = key_embedding / k_len
    k = tf.reduce_sum(key_embedding, 2)
    
    # [batch_sz, mem_sz, embedding_sz]
    v = val_embedding

    # R Matrices
    self.R_list = []
    self.Rb_list = []
    for h in range(FLAGS.hops):
      R_h = tf.get_variable('R_{}'.format(h),
                            [FLAGS.embedding_size, FLAGS.embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer())
      Rb_h = tf.get_variable('Rb_{}'.format(h),
                             [FLAGS.embedding_size],
                             initializer=tf.contrib.layers.xavier_initializer())
      self.R_list.append(R_h)
      self.Rb_list.append(Rb_h)

    # [batch_sz, embedding_sz]
    o = self.key_addressing_and_value_reading(q, k, v)

    # [ans_vocab_sz, embedding_sz]
    if FLAGS.result_embeddings_separate:
      self.B = tf.concat(
        0,
        [nil_word_slot,
         tf.get_variable(
           'B', shape=[ans_vocab_size - 1, FLAGS.embedding_size],
           initializer=tf.contrib.layers.xavier_initializer())])
    else:
      self.B = self.A

    # [ans_vocab_sz]
    self.B_bias = tf.concat(
      0, [[0.0],
          tf.get_variable(
          'B_bias', shape=[ans_vocab_size - 1],
          initializer=tf.contrib.layers.xavier_initializer())])

    #self.B_bias = tf.zeros([ans_vocab_size], dtype=tf.float32)

    # [batch_sz]
    '''
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      self.ans_b, FLAGS.max_ans_count, FLAGS.num_neg_samples,
      True, ans_vocab_size, distortion=0.5, num_reserved_ids=1,
      unigrams=unigram_freq[1:])
    '''
    if FLAGS.loss is not 'NLL':
      sampled_values = tf.nn.uniform_candidate_sampler(
        self.ans_b, FLAGS.max_ans_count, FLAGS.num_neg_samples,
        True, ans_vocab_size)

    if FLAGS.loss is None or FLAGS.loss is 'sampled_softmax':
      self.loss = tf.nn.sampled_softmax_loss(self.B, self.B_bias, o, self.ans_b,
                                             FLAGS.num_neg_samples,
                                             ans_vocab_size,
                                             num_true=FLAGS.max_ans_count,
                                             sampled_values=sampled_values,
                                             remove_accidental_hits=True)
    elif FLAGS.loss is 'margin':
      self.loss = self.margin_loss(o, sampled_values[0])
    elif FLAGS.loss is 'nce':
      self.loss = tf.nn.nce_loss(self.B, self.B_bias, o, self.ans_b,
                                 FLAGS.num_neg_samples,
                                 ans_vocab_size, num_true=FLAGS.max_ans_count,
                                 sampled_values=sampled_values,
                                 remove_accidental_hits=True)
    elif FLAGS.loss is 'NLL':
      self.loss = self.NLL_loss(o, self.neg_samples)
    else:
      assert(False)

    # Sum the loss across the whole batch
    self.loss = tf.reduce_sum(self.loss)

    params = tf.trainable_variables()
    if FLAGS.l2_reg_strength > 0:
      self.regularization_loss = FLAGS.l2_reg_strength * \
          tf.add_n([tf.nn.l2_loss(p) for p in params])
      self.loss = self.loss + self.regularization_loss
    else:
      self.regularization_loss = tf.constant(0)

    #
    # Predictions
    #

    # [embedding_sz, ans_vocab_sz]
    B_t = tf.transpose(self.B, [1, 0])
    # [batch_sz, ans_vocab_sz]
    logits = tf.matmul(o, B_t) + tf.expand_dims(self.B_bias, 0)
    self.pred = tf.argmax(logits, 1)

    # Gradients
    if not forward_only:
      if FLAGS.optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer(FLAGS.initial_learning_rate)
      elif FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)
      elif FLAGS.optimizer == 'gd':
        opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)
      else:
        assert(False)

      grads_and_vars = opt.compute_gradients(self.loss, params)
      if FLAGS.loss is not 'NLL':
        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_gradient_norm), va)
                          for g, va in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), va)
                          for g, va in grads_and_vars]

      self.update = opt.apply_gradients(grads_and_vars, name='train_op')

      if FLAGS.normalize_embeddings:
        self.normalize_A = self.A_var.assign(tf.nn.l2_normalize(self.A_var, 1))

    self.saver = tf.train.Saver(tf.global_variables())

    # Debugging
    self.debug_dict['e'] = (query_embedding, key_embedding, val_embedding)
    self.debug_dict['reduced_e'] = (q, k, v)
    self.debug_dict['final_o'] = o
    self.debug_dict['final_logits'] = logits

  # q_b: [batch_sz, embedding_sz]
  # k_b: [batch_sz, mem_sz, embedding_sz]
  # v_b: [batch_sz, mem_sz, embedding_sz]
  # w_b: [batch_sz, mem_sz]
  # _b is for batch
  def key_addressing_and_value_reading(self, q_b, k_b, v_b):

    # Debugging
    logits_list = [None] * FLAGS.hops
    probs_list = [None] * FLAGS.hops
    o_list = [None] * FLAGS.hops
    q_list = [None] * FLAGS.hops
    self.debug_dict['logits'] = logits_list
    self.debug_dict['probs'] = probs_list
    self.debug_dict['o_list'] = o_list
    self.debug_dict['q_list'] = q_list
    for h in range(FLAGS.hops):

      #
      # Key Addressing
      #

      # [batch_sz, embedding_sz, 1]
      q_temp = tf.expand_dims(q_b, -1)

      # [batch_sz, mem_sz, 1]
      logits = tf.batch_matmul(k_b, q_temp)
      # [batch_sz, mem_sz]
      logits = tf.squeeze(logits)
      probs = tf.nn.softmax(logits)
      # Ignore memory padding
      probs = probs * self.mem_wts_b
      # [batch_sz, 1]
      z = tf.expand_dims(tf.reduce_sum(probs, 1), -1)
      # [batch_sz, mem_sz]
      probs = probs / z

      #
      # Value Reading
      #

      # [batch_sz, mem_sz, 1]
      probs = tf.expand_dims(probs, -1)
      # [batch_sz, embedding_sz]
      o = tf.reduce_sum(probs * v_b, 1)
      R = self.R_list[h]
      R_b = self.Rb_list[h]

      if FLAGS.value_reading == 'o':
        q_b = o
      elif FLAGS.value_reading == 'o.R':
        q_b = tf.matmul(o, R)
      elif FLAGS.value_reading == 'q + o':
        q_b = q_b + o
      elif FLAGS.value_reading == 'q + o.R':
        q_b = q_b + tf.matmul(o, R)
      elif FLAGS.value_reading == 'q.R + o':
        q_b = tf.matmul(q_b, R) + o
      elif FLAGS.value_reading == '(q + o).R':
        if h < FLAGS.hops - 1:
          q_b = tf.matmul(q_b + o, R) + R_b
        else:
          q_b = q_b + o
          # q_b = tf.matmul(q_b + o, R) + R_b
      elif FLAGS.value_reading is None or \
           FLAGS.value_reading == '(q + o).R & o.R':
        if h < FLAGS.hops - 1:
          q_b = tf.matmul(q_b + o, R)
        else:
          q_b = tf.matmul(o, R)
      else:
        assert(False)

      # Debugging
      logits_list[h] = logits
      probs_list[h] = probs
      o_list[h] = o
      q_list[h] = q_b
      # q_b = tf.tanh(q_b)
      
    return q_b

  def margin_loss(self, o, neg_samples):
    # [batch_sz, max_ans_count, embedding_sz]
    pos_e = tf.nn.embedding_lookup(self.B, self.ans_b)
    # [batch_sz, max_ans_count]
    pos_b = tf.nn.embedding_lookup(tf.expand_dims(self.B_bias, -1), self.ans_b)
    pos_b = tf.reshape(pos_b, [-1, FLAGS.max_ans_count])
    # [batch_sz, 1, embedding_sz]
    o_expanded = tf.expand_dims(o, 1)
    # [batch_sz, max_ans_count]
    pos_score = tf.reduce_sum(o_expanded * pos_e, 2) + pos_b
    # [batch_sz, max_ans_count, 1]
    pos_score = tf.expand_dims(pos_score, -1)

    # [FLAGS.num_neg_samples, embedding_sz]
    neg_e = tf.nn.embedding_lookup(self.B, neg_samples)
    # [1, FLAGS.num_neg_samples, embedding_sz]
    neg_e = tf.expand_dims(neg_e, 0)
    # [FLAGS.num_neg_samples]
    neg_b = tf.nn.embedding_lookup(tf.expand_dims(self.B_bias, -1),
                                   neg_samples)
    neg_b = tf.reshape(neg_b, [FLAGS.num_neg_samples])
    # [batch_sz, FLAGS.num_neg_samples]
    neg_score = tf.reduce_sum(o_expanded * neg_e, 2) + neg_b
    # [batch_sz, 1, FLAGS.num_neg_samples]
    neg_score = tf.expand_dims(neg_score, 1)
    loss = tf.maximum(0., 0.1 - pos_score + neg_score)
    return loss

  def NLL_loss(self, o, neg_samples):
    # [batch_sz, max_ans_count, embedding_sz]
    pos_e = tf.nn.embedding_lookup(self.B, self.ans_b)
    # [batch_sz, max_ans_count]
    pos_b = tf.nn.embedding_lookup(tf.expand_dims(self.B_bias, -1), self.ans_b)
    pos_b = tf.reshape(pos_b, [-1, FLAGS.max_ans_count])
    # [batch_sz, 1, embedding_sz]
    o_expanded = tf.expand_dims(o, 1)
    # [batch_sz, max_ans_count]
    pos_logit = tf.reduce_sum(o_expanded * pos_e, 2) + pos_b

    # [batch_sz, FLAGS.num_neg_samples, embedding_sz]
    neg_e = tf.nn.embedding_lookup(self.B, neg_samples)
    # [batch_size, FLAGS.num_neg_samples]
    neg_b = tf.nn.embedding_lookup(tf.expand_dims(self.B_bias, -1),
                                   neg_samples)
    neg_b = tf.reshape(neg_b, [-1, FLAGS.num_neg_samples])
    # [batch_sz, FLAGS.num_neg_samples]
    neg_logits = tf.reduce_sum(o_expanded * neg_e, 2) + neg_b

    # [batch_sz, max_ans_count + FLAGS.num_neg_samples]
    logits = tf.concat(1, [pos_logit, neg_logits])

    # [batch_sz]
    labels = tf.zeros([FLAGS.batch_size], dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    return loss

  def normalize_embeddings(self, session):
    if FLAGS.normalize_embeddings:
      session.run(self.normalize_A)

  def step(self, session, q_inputs, q_lens, ans_inputs, neg_samples,
           mem_key_inputs, key_lens, mem_val_inputs, mem_wts, forward_only,
           test_only=False):
    input_feed = {}
    input_feed[self.q_b] = q_inputs
    input_feed[self.q_len_b] = q_lens
    if not test_only:
      input_feed[self.ans_b] = ans_inputs
    if FLAGS.loss == 'NLL':
      input_feed[self.neg_samples] = neg_samples
    input_feed[self.mem_key_b] = mem_key_inputs
    input_feed[self.key_len_b] = key_lens
    input_feed[self.mem_val_b] = mem_val_inputs
    input_feed[self.mem_wts_b] = mem_wts

    '''
    k_o, k_len_o, kk_o = session.run([self.mem_key_b, self.key_len_b,
                                      self.kk],
                                     input_feed)
    print k_len_o[20][:5]
    print k_o[20][:5][:10]
    print kk_o[20][:5][:10]
    '''
    
    if not forward_only:
      output_feed = [self.regularization_loss, self.loss, self.update]
      self.normalize_embeddings(session)
    elif not test_only:
      output_feed = [self.regularization_loss, self.loss, self.pred]
    else:
      output_feed = [self.pred]

    outputs = session.run(output_feed, input_feed)

    return outputs


def generate_tokens(data_idx_set):
  # Get the # of batches for each bucket and create a turn token for each
  # batch.  The token identifies the batch's bucket and batch # within the
  # bucket.
  tokens = []
  data_set_sz = len(data_idx_set)
  num_batches = data_set_sz / FLAGS.batch_size
  for batch_id in range(num_batches):
      tokens.append(batch_id)
  return tokens


def shuffle_data_idxs_and_generate_tokens(data_idx_set):
  # Shuffle the data indices
  np.random.shuffle(data_idx_set)
  tokens = generate_tokens(data_idx_set)
  return tokens


def prepare_batch(data, batch_num, data_idx_set, batch_size,
                  ans_vocab_size, ignore_words=None):
  start_idx = batch_num * batch_size
  v_qa_pairs, q_to_kv, vectorized_windows = data
  batch_q_nums = []
  batch_q_lens = []
  batch_m_keys = []
  batch_key_lens = []
  batch_m_vals = []
  batch_qs = []
  batch_ans = []
  batch_neg_samples = []
  batch_wts = []
  win_sz = len(vectorized_windows[(0, 1)][0])
  pad_win = [0] * win_sz

  for offset in xrange(batch_size):
    q_num = data_idx_set[start_idx + offset]
    batch_q_nums.append(q_num)
    q, anslist, q_len = v_qa_pairs[q_num]
    if ignore_words:
      ignored = [t for t in q if t in ignore_words]
      q = [t if t not in ignore_words else 0 for t in q]
      q_len = q_len - len(ignored)

    batch_qs.append(q)
    assert(np.sqrt(q_len) > 0)
    batch_q_lens.append(np.sqrt(q_len))

    replace = True if FLAGS.max_ans_count > len(anslist) else False
    ans = np.random.choice(anslist, FLAGS.max_ans_count, replace=replace)
    batch_ans.append(ans)

    m_keys = []
    m_vals = []
    k_lens = []
    wts = []

    if FLAGS.loss == 'NLL':
      # IGNORE the pad word at the start of the dictionary
      '''
      neg_samples = np.random.choice(vocab_idxs,
      FLAGS.num_neg_samples * 2,
      replace=False)
      '''
      while True:
        samples = set([np.random.randint(1, ans_vocab_size + 1)
                       for i in range(FLAGS.num_neg_samples +
                                      100 + len(anslist))])
        neg_samples = samples - set(anslist)
        if len(neg_samples) >= FLAGS.num_neg_samples:
          break

      neg_samples = list(neg_samples)[:FLAGS.num_neg_samples]
      batch_neg_samples.append(neg_samples)

    for story_idx, pos in list(q_to_kv[q_num])[:FLAGS.mem_size]:
      k, v, k_len = vectorized_windows[(story_idx, pos)]
      if ignore_words:
        ignored = [t for t in k if t in ignore_words]
        k = [t if t not in ignore_words else 0 for t in k]
        k_len = k_len - len(ignored)

      m_keys.append(k)
      m_vals.append(v)
      assert(np.sqrt(k_len) > 0)
      k_lens.append(np.sqrt(k_len))
      wts.append(1)

    pad_len = FLAGS.mem_size - len(m_keys)
    m_keys += [pad_win] * pad_len
    k_lens += [0] * pad_len
    m_vals += [0] * pad_len
    wts += [0.0] * pad_len
    batch_m_keys.append(m_keys)
    batch_m_vals.append(m_vals)
    batch_key_lens.append(k_lens)
    batch_wts.append(wts)

  tup = (batch_q_nums,
         np.array(batch_qs, dtype=np.int32),
         np.array(batch_q_lens, dtype=np.float32).reshape(-1, 1),
         np.array(batch_ans, dtype=np.int64),
         np.array(batch_neg_samples, dtype=np.int32),
         np.array(batch_m_keys, dtype=np.int32),
         np.array(batch_key_lens, dtype=np.float32).reshape(-1, FLAGS.mem_size, 1),
         np.array(batch_m_vals, dtype=np.int32),
         np.array(batch_wts, dtype=np.float32))
  return tup


def print_batch(data, data_set_idxs, batch_num, preds, answers,
                rev_vocab, rev_ans_vocab):
  v_qa_pairs, q_to_kv, vectorized_windows = data
  start_idx = batch_num * FLAGS.batch_size
  for offset in xrange(FLAGS.batch_size):
    q_num = data_set_idxs[start_idx + offset]
    q, ans, _ = v_qa_pairs[q_num]
    print '=> Question #:', q_num
    print ' '.join([rev_vocab[t] for t in q if t != 0])
    print '\n'.join([rev_ans_vocab[t] for t in ans if t != 0])
    print "**Predicted Answer**:", rev_ans_vocab[preds[offset]]
    if (preds[offset] in list(answers[offset])):
      print "--correct prediction--"


def run_batches(session, model, data, shuffled_idxs, forward_only, tokens,
                test_only=False):
  v_qa_pairs, _, _ = data
  
  # Run the epoch by looping over the tokens
  loss = 0.0
  reg_loss_total = 0.0
  correct_predictions = 0

  if FLAGS.result_embeddings_separate:
    ans_vocab = data_dict['ans_vocab']
  else:
    ans_vocab = data_dict['vocab']
  ans_vocab_size = len(ans_vocab)

  bad_q_nums = []
  for batch_num in tokens:
    q_nums, q_inputs, q_lens, ans_inputs, neg_samples, key_inputs, key_lens,\
      value_inputs, wts = \
      prepare_batch(data, batch_num, shuffled_idxs, FLAGS.batch_size,
                    ans_vocab_size)

    if not test_only:
      reg_loss, step_loss, outputs = model.step(session,
                                                q_inputs, q_lens,
                                                ans_inputs, neg_samples,
                                                key_inputs, key_lens,
                                                value_inputs, wts,
                                                forward_only, test_only)
      loss += step_loss
      reg_loss_total += reg_loss
    else:
      outputs = model.step(session,
                           q_inputs, q_lens, None, None,
                           key_inputs, key_lens, value_inputs,
                           wts, forward_only, test_only)
      outputs = outputs[0]
      if FLAGS.mode == 'test_debug':
        rev_vocab = model.data_dict['rev_vocab']
        rev_ans_vocab = model.data_dict['rev_vocab']
        print_batch(data, shuffled_idxs, batch_num, outputs,
                    ans_inputs, rev_vocab, rev_ans_vocab)

    if not forward_only:
      continue

    for i, q_num in enumerate(q_nums):
      _, anslist, _ = v_qa_pairs[q_num]
      if (outputs[i] in set(anslist)):
        correct_predictions += 1
      else:
        bad_q_nums.append(q_num)

  avg_loss = loss / len(tokens)
  reg_loss_avg = reg_loss_total / len(tokens)
  instance_count = len(tokens) * FLAGS.batch_size
  accuracy = 1.0 * correct_predictions / instance_count
  return (reg_loss_avg, avg_loss, accuracy,
          correct_predictions, instance_count, bad_q_nums)


def create_model(session, forward_only, data_dict):
  # Create model.
  print("Creating Key-Value End-to-End Memory Network Model")
  model = KVMemN2N_Model(forward_only, data_dict)

  # Merge all the summaries and write them out to /tmp/train (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/train',
                                        graph=session.graph)
  #test_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/test')

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    model.normalize_embeddings(session)

  return model


def data_set_str_to_idx(name):
  return {'train': 0, 'dev': 1, 'test': 2}[name]


def q_num_find(q_str, data_dict, data_set_str='train'):
  def text_to_vector(vocab, txt):
    return [vocab[t] for t in txt]

  tokenized_q = ['1:' + item.strip().lower() for item in q_str.split('1:')
                 if item is not u'']
  vocab = data_dict['vocab']
  set_idx = data_set_str_to_idx(data_set_str)
  v_qa_pairs = data_dict['v_qa_pairs_data'][set_idx]
  max_q_len = data_dict['max_q_len']
  pad = [0] * (max_q_len - len(tokenized_q))
  v_q = text_to_vector(vocab, tokenized_q) + pad
  for qnum, (q, a, _) in v_qa_pairs.items():
    if q == v_q:
      return qnum
  return None


def train(data_dict):

  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = create_model(session, False, data_dict)
    v_qa_pairs_train, v_qa_pairs_dev, _ = data_dict['v_qa_pairs_data']
    q_to_kv_train, q_to_kv_dev, _ = data_dict['q_to_kv_maps']
    vectorized_windows = data_dict['vectorized_windows']
    train_set = (v_qa_pairs_train, q_to_kv_train, vectorized_windows)
    dev_set = (v_qa_pairs_dev, q_to_kv_dev, vectorized_windows)
    train_set_idxs = list(set(v_qa_pairs_train.keys()) &
                          set(q_to_kv_train.keys()))
    dev_set_idxs = list(set(v_qa_pairs_dev.keys()) & set(q_to_kv_dev.keys()))

    # This is the training loop.
    for epoch in range(500):

      print 'Epoch: ', epoch
      tokens = shuffle_data_idxs_and_generate_tokens(train_set_idxs)
      i = 0
      while True:
        start_time = time.time()
        reg_loss, loss, _, _, _, _ = run_batches(session, model, train_set,
                                                 train_set_idxs,
                                                 False, tokens[i:i + 500])
        run_time = time.time() - start_time

        # Print statistics for the previous epoch.
        print("run-time %.2f cross-entropy loss %.2f reg-loss %.2f" %
              (run_time, loss, reg_loss))
        i += 500

        dev_tokens = shuffle_data_idxs_and_generate_tokens(dev_set_idxs)
        reg_loss, loss, acc, cps, n, _ = run_batches(session, model, dev_set,
                                                     dev_set_idxs,
                                                     True, dev_tokens[:20])
        print("  eval: reg-loss %.2f loss %.2f acc %.2f"
              " correct preds %d insts %d" %
              (reg_loss, loss, acc, cps, n))

        if i >= len(tokens):
          break

      # Save checkpoint and zero timer and loss.
      checkpoint_path = os.path.join(FLAGS.train_dir, "movieqa.ckpt")
      model.saver.save(session, checkpoint_path)

      # Print statistics for the previous epoch.
      # print "epoch-time %.2f cross-entropy loss %.2f" % (epoch_time, loss)
      dev_tokens = shuffle_data_idxs_and_generate_tokens(dev_set_idxs)
      reg_loss, loss, acc, cps, n, _ = run_batches(session, model, dev_set,
                                                   dev_set_idxs,
                                                   True, dev_tokens)
      print("  eval: reg-loss %.2f loss %.2f acc %.2f"
            " correct preds %d insts %d" %
            (reg_loss, loss, acc, cps, n))

      # Run evals on development set and print their perplexity.
      # eval_loss, accuracy = run_epoch(sess, model, dev_set, True)
      # print("  eval: loss %.2f, accuracy %.2f" % (eval_loss, accuracy))
      sys.stdout.flush()


def test(data_dict, batch_count=None, data_set_str='test', q_str=None):
  set_idx = data_set_str_to_idx(data_set_str)
  v_qa_pairs = data_dict['v_qa_pairs_data'][set_idx]
  q_to_kv_map = data_dict['q_to_kv_maps'][set_idx]
  vectorized_windows = data_dict['vectorized_windows']
  data_set = (v_qa_pairs, q_to_kv_map, vectorized_windows)
  if q_str:
    qnum = q_num_find(q_str, data_dict, data_set_str)
    data_set_idxs = [qnum]
    FLAGS.batch_size = 1
  else:
    data_set_idxs = list(set(v_qa_pairs.keys()) & set(q_to_kv_map.keys()))

  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = create_model(session, True, data_dict)
    tokens = shuffle_data_idxs_and_generate_tokens(data_set_idxs)
    tokens = tokens[:batch_count] if batch_count else tokens
    _, _, acc, cps, n, bad_q_nums = run_batches(session, model, data_set,
                                                data_set_idxs,
                                                True, tokens, test_only=True)
    print("  Test: accuracy %.2f, correct predictions %d,"
          " instances %d" %
          (acc, cps, n))
    sys.stdout.flush()

  return bad_q_nums

def train_one_q(data_dict, q_str, iters):
  qnum = q_num_find(q_str, data_dict, data_set_str='train')
  set_idx = data_set_str_to_idx('train')
  v_qa_pairs = data_dict['v_qa_pairs_data'][set_idx]
  q_to_kv_map = data_dict['q_to_kv_maps'][set_idx]
  vectorized_windows = data_dict['vectorized_windows']
  data_set = (v_qa_pairs, q_to_kv_map, vectorized_windows)
  data_set_idxs = [qnum]
  FLAGS.batch_size = 1

  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = create_model(session, False, data_dict)

    tokens = shuffle_data_idxs_and_generate_tokens(data_set_idxs)
    for i in range(iters):
      reg_loss, loss, _, _, _ = run_batches(session, model, data_set,
                                            data_set_idxs,
                                            False, tokens, test_only=False)
      if i % 1000 == 0: print i

    # Save checkpoint and zero timer and loss.
    checkpoint_path = os.path.join(FLAGS.train_dir, "movieqa.ckpt")
    model.saver.save(session, checkpoint_path)


if __name__ == "__main__":
  print FLAGS.train_dir
  data_dict = data_utils.prepare_data(FLAGS)
  if FLAGS.mode == 'test':
    test(data_dict, batch_count=None, data_set_str='test')
  elif FLAGS.mode == 'test_debug':
    q_str = u'1:who 1:is 1:the 1:creator 1:of 1:the 1:film 1:script 1:for 1:Kitty Foyle 1:?'
    test(data_dict, batch_count=None, data_set_str='dev', q_str=q_str)
  elif FLAGS.mode == 'train_1_q':
    q_str = u'1:what 1:movie 1:did 1:Christopher Morley 1:write 1:the story 1:for 1:?'
    train_one_q(data_dict, q_str, 1000)
  elif FLAGS.mode is None or FLAGS.mode == 'train':
    print "Learning rate: ", FLAGS.initial_learning_rate
    train(data_dict)
  else:
    assert(False)
