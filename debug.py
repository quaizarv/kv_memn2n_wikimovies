#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import collections
import train
import data_utils


def show_kv(qnum, data_set, rev_vocab, mem_keys, mem_vals, mem_idxs=None):
  def vector_to_text(v):
    return [rev_vocab[t] for t in v if t != 0]

  v_qa_pairs, q_to_kv_map, vectorized_windows = data_set
  kv_pairs = list(q_to_kv_map[qnum])
  if mem_idxs is None:
    mem_idxs = range(5)
  print u'  Window Positions:'
  print u'   ', [kv_pairs[mem_idx] for mem_idx in mem_idxs]
  for mem_idx in mem_idxs:
    k, v = mem_keys[mem_idx], mem_vals[mem_idx]
    # k, v = vectorized_windows[win_idx]
    print u'  ', k, v
    print u'  ', vector_to_text(k), rev_vocab[v]


def show_qa(qnum, data_set, rev_vocab):
  def vector_to_text(v):
    return [rev_vocab[t] for t in v if t != 0]

  v_qa_pairs, q_to_kv_map, vectorized_windows = data_set
  q, a = v_qa_pairs[qnum]
  print qnum
  print q
  print vector_to_text(q)
  print a
  print vector_to_text(a)
  kv_pairs = list(q_to_kv_map[qnum])
  print u'Relevant Memory Count: ', len(kv_pairs)


def debug_question(session, model, qnum, data_set_str, w_freq_stats):
  w_d_freq, w_val_freq, w_q_freq, w_ans_freq = w_freq_stats
  data_dict = model.data_dict
  set_idx = train.data_set_str_to_idx(data_set_str)
  v_qa_pairs = data_dict['v_qa_pairs_data'][set_idx]
  q_to_kv_map = data_dict['q_to_kv_maps'][set_idx]
  vectorized_windows = data_dict['vectorized_windows']
  rev_vocab = data_dict['rev_vocab']

  data_set = (v_qa_pairs, q_to_kv_map, vectorized_windows)
  _, q_inputs, ans_inputs, key_inputs, value_inputs, wts = \
      train.prepare_batch(data_set, 0, [qnum], 1)
  show_qa(qnum, data_set, rev_vocab)

  input_feed = {}
  input_feed[model.q_b] = q_inputs
  input_feed[model.ans_b] = ans_inputs
  input_feed[model.mem_key_b] = key_inputs
  input_feed[model.mem_val_b] = value_inputs
  input_feed[model.mem_wts_b] = wts

  debug_obj_names = model.debug_dict.keys()
  debug_objs = model.debug_dict.values()
  results = session.run(debug_objs, input_feed)
  debug_results = {k: r for k, r in zip(debug_obj_names, results)}

  q_e, k_e, v_e = debug_results['reduced_e']
  # logits = debug_results['logits']
  probs = debug_results['probs']
  o_list = debug_results['o_list']
  q_list = debug_results['q_list']
  final_logits = debug_results['final_logits']

  print u"Q norm: ", np.linalg.norm(q_e[0])
  print u'\n==Matching key-values=='
  for h in range(train.FLAGS.hops):
    o = o_list[h]
    q = q_list[h]
    print u'Hop: ', h
    best = sorted(enumerate(probs[h][0]), key=lambda (i, p): -p)
    for i, p in best[:5]:
      word_idx = value_inputs[0][i]
      if word_idx == 0:
        continue
      print u'  {0}: mem-index {1}, prob {2}, norm {3}'.format(
        rev_vocab[word_idx], i, p, np.linalg.norm(v_e[0][i]))
      print u'    doc-freq {0}, val-freq {1}, q-freq {2}, ans-freq {3}'.format(
        w_d_freq[word_idx], w_val_freq[word_idx],
        w_q_freq[word_idx], w_ans_freq[word_idx],)
      show_kv(qnum, data_set, rev_vocab,
              key_inputs[0], value_inputs[0], mem_idxs=[i])
      print u'    logit-final {0}'.format(final_logits[0][word_idx])
      print
    print u'  o norm', np.linalg.norm(o)
    print u'  q norm', np.linalg.norm(q)
    print u'  cosine_similarity(o, q)', cosine_similarity(o, q)

  print u'\n==Best Answers=='
  best_answers = sorted(enumerate(final_logits[0]), key=lambda (i, l): -l)
  for i, l in best_answers[:5]:
    if i == 0:
      continue
    print u'  {0}: dict-index {1}, logit(q,w) {2}   '.format(
      rev_vocab[i], i, l)


def w_freq_stats(data_dict):
  v_wins = data_dict['vectorized_windows']
  w_doc_freq = collections.Counter([t for k in v_wins.keys()
                                    for t in v_wins[k][0]])
  w_val_freq = collections.Counter([win[1] for win in v_wins.values()])

  v_qa_pairs, _, _ = data_dict['v_qa_pairs_data']
  w_q_freq = collections.Counter([t for q, a in v_qa_pairs.values()
                                  for t in q])

  w_ans_freq = collections.Counter([t for q, a in v_qa_pairs.values()
                                    for t in a])
  return (w_doc_freq, w_val_freq, w_q_freq, w_ans_freq)


def debug_questions(data_dict, q_num_set):
  w_f_stats = w_freq_stats(data_dict)
  tf.reset_default_graph()
  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = train.create_model(session, True, data_dict)
    for qn in q_num_set:
      debug_question(session, model, qn, 'dev', w_f_stats)
      print(u'=' * 80)


def w_freq_stats_sorted(data_dict):
  tf.reset_default_graph()
  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = train.create_model(session, True, data_dict)
    A = session.run([model.A])[0]

  w_d_freq, w_val_freq, w_q_freq, w_ans_freq = w_freq_stats(data_dict)
  vocab = data_dict['vocab']
  w_doc_freq_sorted = sorted([(w_d_freq[vocab[k]], k,
                               np.linalg.norm(A[vocab[k]]))
                              for k in vocab.keys()], reverse=True)
  w_val_freq_sorted = sorted([(w_val_freq[vocab[k]], k,
                               np.linalg.norm(A[vocab[k]]))
                              for k in vocab.keys()], reverse=True)
  w_q_freq_sorted = sorted([(w_q_freq[vocab[k]], k,
                             np.linalg.norm(A[vocab[k]]))
                            for k in vocab.keys()], reverse=True)
  w_ans_freq_sorted = sorted([(w_ans_freq[vocab[k]], k,
                               np.linalg.norm(A[vocab[k]]))
                              for k in vocab.keys()], reverse=True)

  return (w_doc_freq_sorted, w_val_freq_sorted,
          w_q_freq_sorted, w_ans_freq_sorted)


def freq_trained_words(data_dict):
  tf.reset_default_graph()
  with tf.Session() as session:
    # with tf.device('/cpu:0'):
    model = train.create_model(session, True, data_dict)
    A = session.run([model.A])[0]

  w_d_freq, w_val_freq, w_q_freq, w_ans_freq = w_freq_stats(data_dict)
  vocab = data_dict['vocab']
  results = sorted([(np.linalg.norm(A[i]), w_d_freq[i], w_val_freq[i],
                     w_q_freq[i], w_ans_freq[i], k)
                    for k, i in vocab.items()
                    if (w_ans_freq[i] + w_q_freq[i] > 5 or
                        w_val_freq[i] + w_d_freq[i] > 50)],
                   reverse=False)
  return zip(*results)[-1][1:]


if __name__ == "__main__":
  data_dict = data_utils.prepare_data(train.FLAGS)
  bad_q_nums = train.test(data_dict, batch_count=None, data_set_str='dev')
  bad_qn_sample = np.random.choice(bad_q_nums, 100, replace=False)
  debug_questions(data_dict, bad_qn_sample)
  good_q_nums = set(data_dict['v_qa_pairs_data'][1].keys()) - set(bad_q_nums)
  good_qn_sample = np.random.choice(list(good_q_nums), 100, replace=False)
  debug_questions(data_dict, good_qn_sample)
  
