import os
import time
import re
import codecs
import itertools as itools
import collections
import pickle
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def timeit(orig_fn):
    def new_fn(*args, **kwargs):
        t = time.time()
        ret = orig_fn(*args, **kwargs)
        print('time = {} '.format(time.time() - t))
        return ret
    return new_fn


@timeit  
def parse_docs(fname):

  def read_docs(fname):
    wiki_file = codecs.open(fname, encoding='utf-8')
    lines = [line for line in wiki_file]
    lines = [line.strip() for line in lines]
    lines = u'\n'.join(lines)
    docs = lines.split(u'\n\n')
    return docs

  def process_line(line, doc, doc_num):
    line = line.strip()
    key, val = line.split(u'\t')
    key_tokens = key.split(u' ')
    try:
      pos_in_doc = int(key_tokens[0])
    except ValueError:
      print doc
      print doc_num
      raise
    key_type = key_tokens[1]
    key = u' '.join(key_tokens[2:])
    key = [tok.strip() for tok in re.findall('\d:.*?(?=\d:|$)', key,
                                             flags=re.U)]
    key_toks = []
    for tok in key:
      if tok[:2] == u'0:':
        key_toks.append(u'1:' + tok[2:] if tok[2:] != val else tok[2:])
      else:
        key_toks.append(tok)
    return (pos_in_doc, [key_type] + key_toks, val)

  docs = read_docs(fname)
  docs = [[process_line(line, doc, doc_num) for line in doc.split(u'\n')
           if len(line) > 0]
          for doc_num, doc in enumerate(docs) if len(doc) > 0]
  return docs


@timeit
def vocab_compute(docs):
  vocab = {}
  ans_vocab = {}
  word_freq = {}
  for doc_num, doc in enumerate(docs):
    for _, tokens, val in doc:
      v = val.lower()
      ans_vocab[v] = 1
      word_freq[v] = word_freq.get(v, 0) + 1
      for tok in tokens:
        t = tok.lower()
        if tok[:2] == '1:':
          vocab[t] = 1
        else:
          ans_vocab[t] = 1
        word_freq[t] = word_freq.get(t, 0) + 1

  ans_vocab = ['_NIL'] + ans_vocab.keys()
  vocab = {k: i for i, k in enumerate(ans_vocab + vocab.keys())}
  ans_vocab = {k: i for i, k in enumerate(ans_vocab)}
  return vocab, ans_vocab, word_freq


def docs_to_films(docs):
  def doc_to_film(doc):
    film_str = doc[0][1][1]
    for _ in range(2):
        if film_str[:2] == u'1:':
            film = film_str[2:]
        else:
            film = film_str
        if film == u'"':
            film_str = doc[0][1][2]
        else:
            break
    return film

  doc_to_film_map = {dn: doc_to_film(doc) for dn, doc in enumerate(docs)}
  return set(doc_to_film_map.values()), doc_to_film_map


@timeit  
def vectorized_windows(FLAGS, docs, vocab, max_sentence_len):
  v_windows = {}
  for doc_idx, doc in enumerate(docs):
    for pos, tokens, val in doc:
      key_v = [vocab[tok.lower()] for tok in tokens if tok.lower() in vocab]
      key_len = len(key_v)
      key_v = key_v + [vocab[FLAGS.pad_word]] * (max_sentence_len - len(key_v))
      val_v = vocab[val.lower()]
      v_windows[(doc_idx, pos)] = (key_v, val_v, key_len)
  return v_windows


@timeit  
def entities(entities_file_name, vocab):
  ent_list = []
  with codecs.open(entities_file_name, encoding='utf-8') as read:
    for l in read:
      l = l.strip()
      if len(l) > 0 and l.lower() in vocab:
        ent_list.append(l)
  ent_list.sort(key=lambda x: -len(x))
  return ent_list


@timeit
def filter_qa_pairs(qa_pairs, films, ans_vocab, low_idf_words):
  '''
    Filtering is done based a reduced set of films
  '''
  filtered_qa_pairs = []
  if films:
    films = [f for f in films
             if len(f) > 2 and f not in low_idf_words]
    films = set(['1:' + f for f in films]) | set(films)

  for q_toks, alist in qa_pairs:
    ans_are_movies = False
    if films:
      ans_are_movies = any(a in films for a in alist)

    alist = [ans for ans in alist
             if (ans.lower() in ans_vocab) and
             (not ans_are_movies or ans in films)]
    if not alist:
      continue

    if not films:
      filtered_qa_pairs.append((q_toks, alist))
      continue

    for t in alist + q_toks:
      if t in films:
        filtered_qa_pairs.append((q_toks, alist))
        break

  return filtered_qa_pairs


def process_qa_pair(q, a, rx):
  m = re.search(u"1:", q)
  # Keep the tokenized question, throwing away the original question at the
  # start
  q = q[m.start():]
  q_toks = [tok.strip() for tok in re.findall('\d:.*?(?=\d:|$)', q)]

  # Now add back the original question
  q_toks = [tok[2:] for tok in q_toks] + q_toks
  
  alist = []
  m = rx.search(a, 0)
  while m:
    alist.append(a[m.start():m.end()])
    m = rx.search(a, m.end())

  return q_toks, alist


@timeit  
def process_qa_file(qa_file_name, ent_list, ans_vocab):
  pat = u'|'.join(u'\\b{}\\b'.format(re.escape(e)) for e in ent_list)
  rx = re.compile(pat, flags=re.UNICODE)
  qa_file = codecs.open(qa_file_name, encoding='utf-8')
  lines = [line.strip() for line in qa_file]
  qa_pairs = [line.split('\t') for line in lines]
  processed_qa_pairs = [process_qa_pair(q, a, rx) for q, a in qa_pairs]
  return processed_qa_pairs


@timeit
def inverted_idx_compute(docs, ent_list, word_freq):
  inverted_idx = {}
  for doc_num, doc in enumerate(docs):
    for win in doc:
      pos, tokens, ans = win
      for tok in tokens:
        t = tok.lower()
        if t not in ent_list and t[2:] not in ent_list:
            continue
        if tok[:2] != u'1:' or t not in word_freq:
          continue
        if word_freq[t] >= 1000:
          continue
        if t not in inverted_idx:
            inverted_idx[t] = []
        inverted_idx[t].append((doc_num, pos))

  inverted_idx = {tok: pl for tok, pl in inverted_idx.items()
                  if len(pl) <= 1000}

  return inverted_idx


def tfidf_models_by_docs(docs):
  def win_preprocessor(win):
    return [t.lower() for t in win]

  def win_tokenizer(win):
    return win

  models = {}
  for dn, doc in enumerate(docs):
    model = TfidfVectorizer(preprocessor=win_preprocessor,
                            tokenizer=win_tokenizer)
    doc_wins = list(zip(*doc)[1])
    X = model.fit_transform(doc_wins)
    models[dn] = (model, X)

  return models


def q_to_matching_docs(q_to_kv_pairs):
    return {q_num: set([dn for dn, pos in list(kv_set)])
            for q_num, kv_set in q_to_kv_pairs.items()}


def corpus_tfidf_model(docs):
  # Aggregate windows in a document to a single list of tokens
  docs_no_win = [list(itools.chain(*[win for _, win, val in doc]))
                 for doc in docs]

  # Convert the document corpus to a tf-idf matrix

  # Convert all tokens to lowercase
  def doc_preprocessor(doc):
    return [t.lower() for t in doc]

  # Document is already tokenized
  def doc_tokenizer(doc):
    return doc
  model = TfidfVectorizer(preprocessor=doc_preprocessor,
                          tokenizer=doc_tokenizer)
  X = model.fit_transform(docs_no_win)
  return (model, X)


def ranked_wins_expand(ranked_matches, docs):
  ranked_wins = []
  covered = {}
  for dn, pos in ranked_matches:
    first = max(pos - 5, 1)
    last = min(pos + 4, len(docs[dn]))
    for p in range(first, last + 1):
      if (dn, p) not in covered:
        ranked_wins.append((dn, p))
        covered[(dn, p)] = 1
  return ranked_wins


def ranked_doc_windows(docs, q, dn, wins, tfidf_models):
  model, X = tfidf_models[dn]
  q_v = model.transform([q])
  sim_matrix = cosine_similarity(X, q_v)
  sim_matrix.shape = (sim_matrix.size,)
  ranking = sorted(((dn, pos) for _, pos in wins),
                   key=lambda (dn, pos): sim_matrix[pos - 1],
                   reverse=True)
  return ranked_wins_expand(ranking, docs)


@timeit
def q_to_best_kv_pairs(FLAGS, docs, corpus_tfidf_model,
                       qa_pairs, q_to_kv_pairs,
                       films, doc_to_film_map,
                       tfidf_models_by_docs):

  model, X = corpus_tfidf_model
  cnt = 0
  q_to_docs = q_to_matching_docs(q_to_kv_pairs)
              
  # For each Question, rank matching docs and then rank matching key-value
  # pairs within docs
  best_kv_pairs_dict = {}
  for q_num, (q, alist) in enumerate(qa_pairs):
    if q_num not in q_to_kv_pairs:
      continue
    kv_pairs = q_to_kv_pairs[q_num]
    key_wins = ranked_wins_expand(kv_pairs, docs)
    if len(key_wins) < FLAGS.mem_size:
      best_kv_pairs_dict[q_num] = key_wins
      continue

    q_v = model.transform([q])
    sim_matrix = cosine_similarity(X, q_v)
    sim_matrix.shape = (sim_matrix.size,)
    ranking = sorted((dn for dn in q_to_docs[q_num]),
                     key=lambda dn: sim_matrix[dn],
                     reverse=True)

    ranking = {dn: i for i, dn in enumerate(ranking)}
    kv_pairs = sorted(((dn, pos) for dn, pos in kv_pairs),
                      key=lambda (dn, pos): ranking[dn])

    best_kv_pairs = []
    for dn, wins in itools.groupby(kv_pairs, key=lambda (dn, pos): dn):
      ranked_wins = ranked_doc_windows(docs, q, dn, wins, tfidf_models_by_docs)
      best_kv_pairs += ranked_wins[:FLAGS.mem_size - len(best_kv_pairs)]
      if len(best_kv_pairs) >= FLAGS.mem_size:
        break

    # If answer is a list of films, ensure that we only keep the ones which
    # have corresponding documents present in the selected key-value pairs
    if any(a in films for a in alist):
      new_alist = []
      dn_set = set([dn for dn, _ in best_kv_pairs])
      for dn in dn_set:
        if doc_to_film_map[dn] in alist:
          new_alist.append(doc_to_film_map[dn])
      if not new_alist:
        continue
      qa_pairs[q_num] = (q, new_alist)

    best_kv_pairs_dict[q_num] = best_kv_pairs
    cnt += 1

  print "count", cnt
  return best_kv_pairs_dict


@timeit  
def q_to_key_value_pairs(qa_pairs, inverted_idx, low_idf_words):
  # stopwords_set = set(stopwords.words('english'))
  q_to_kv_pairs = {}
  for q_num, (q, alist) in enumerate(qa_pairs):
    kv_pairs = []
    for tok in q:
      tok = tok.lower()
      # if tok[2:] in stopwords_set:
      #  continue
      # if tok in low_idf_words:
      #  continue
      if (tok not in inverted_idx):
        continue
      # kv_pairs += (q_word_freq(tok), inverted_idx[tok.lower()])
      kv_pairs += inverted_idx[tok.lower()]
    q_to_kv_pairs[q_num] = kv_pairs

  q_to_kv_pairs = {q_num: set(pl) for q_num, pl in q_to_kv_pairs.items()
                   if len(pl) > 0}
  return q_to_kv_pairs


@timeit
def vectorized_qa_pairs(FLAGS, qa_pairs, q_to_kv_map,
                        vocab, max_q_len):
  vectorized_qa_pairs = {}
  for q_num, (q, ans_list) in enumerate(qa_pairs):
    if q_num not in q_to_kv_map:
      continue
    q_v = [vocab[tok.lower()] for tok in q if tok.lower() in vocab]
    q_len = len(q_v)
    q_v = q_v + [vocab[FLAGS.pad_word]] * (max_q_len - len(q_v))
    ans_v = [vocab[a.lower()] for a in ans_list if a.lower() in vocab]
    if len(ans_v) == 0:
      continue
    '''
    np.random.shuffle(ans_v)
    ans_v = ans_v[:max_ans_count]
    if len(ans_v) < max_ans_count:
      remaining = max_ans_count - len(ans_v)
      ans_v = ans_v + (ans_v * (1 + (remaining - 1) / len(ans_v)))
      ans_v = ans_v[:max_ans_count]
    '''
    vectorized_qa_pairs[q_num] = (q_v, ans_v, q_len)
  return vectorized_qa_pairs


@timeit
def prepare_data(FLAGS):

  def step():
    i = 0
    while True:
      i += 1
      yield i

  processed_data_file = os.path.join(FLAGS.data_dir, FLAGS.data_pkl_file)
  if os.path.exists(processed_data_file):
    with open(processed_data_file, 'rb') as pkl_file:
      data_dict = pickle.load(pkl_file)
    return data_dict

  step = step()

  print('Parsing {}. {}'.format(step.next(), FLAGS.win_file))
  docs = parse_docs(os.path.join(FLAGS.data_dir, FLAGS.win_file))

  print('{}. Calculating Vocabulary'.format(step.next()))
  vocab, ans_vocab, word_freq = vocab_compute(docs)

  print('{}. Reading Entities'.format(step.next()))
  ent_list = entities(os.path.join(FLAGS.input_dir, FLAGS.entities_file),
                      vocab)

  # Never compute entities on a reduced vocab
  if FLAGS.docs_count and FLAGS.docs_count < len(docs):
    print('{}. Reducing Corpus & Vocabulary'.format(step.next()))
    kitty_foyle = [doc for doc in docs
                   if doc[0][1][1].lower() == u'kitty foyle' or
                   doc[0][1][1][2:].lower() == u'kitty foyle']
    docs = list(np.random.choice(docs, FLAGS.docs_count, replace=False))
    docs = docs[:-1] + kitty_foyle
    vocab, ans_vocab, _ = vocab_compute(docs)

  for t in [u'1:and', u'1:.']:
    del vocab[t]

  max_sentence_len = max(len(tokens) for doc in docs for _, tokens, val in doc)
  films, doc_to_film_map = docs_to_films(docs)

  print('{}. Vectorizing Window based  Key/Value pairs'.format(step.next()))
  v_wins = vectorized_windows(FLAGS, docs, vocab, max_sentence_len)

  print('{}. Reading QA Files'.format(step.next()))
  qa_pairs_data = [process_qa_file(os.path.join(FLAGS.data_dir, fname),
                                   ent_list, ans_vocab)
                   for fname in ['train_1.txt', 'dev_1.txt', 'test_1.txt']]

  qa_pairs_data_train = qa_pairs_data[0]

  qa_word_freq = collections.Counter(
      itools.chain(*[q_toks + alist for q_toks, alist in qa_pairs_data_train]))
  for t in qa_word_freq:
      word_freq[t] = word_freq.get(t, 0) + qa_word_freq[t]
  q_word_freq = collections.Counter(
      itools.chain(*[q_toks for q_toks, alist in qa_pairs_data_train]))
  low_idf_words = sorted([(f, k) for k, f in q_word_freq.items()],
                         reverse=True)[:100]
  low_idf_words = [k for f, k in low_idf_words]

  if FLAGS.docs_count and FLAGS.docs_count < len(docs):
      qa_pairs_data = [filter_qa_pairs(qa_pairs, films,
                                       ans_vocab, low_idf_words)
                       for qa_pairs in qa_pairs_data]

  max_q_len = max(max(len(q) for q, _ in qa_pairs)
                  for qa_pairs in qa_pairs_data)

  print('{}. Calculating Inverted Index'.format(step.next()))
  ent_list_lower = set(t.lower() for t in ent_list)
  inverted_idx = inverted_idx_compute(docs, set(ent_list_lower), word_freq)

  # Update the vocab to include words which are present in questions but not in
  # the documents. This is needed only when we have are working with a
  # restricted document set
  if FLAGS.docs_count and FLAGS.docs_count < len(docs):
    q_vocab = set(itools.chain(*(q for q, a in qa_pairs_data[0])))
    q_vocab = set(t for t in q_vocab
                  if t not in vocab and t.lower() not in vocab)
    vocab.update((t.lower(), i)
                 for i, t in enumerate(list(q_vocab), len(vocab)))

  rev_vocab = {i: k for k, i in vocab.items()}
  rev_ans_vocab = {i: k for k, i in ans_vocab.items()}

  print('{}. Computing Question to Key Value Pairs Mapping'.format(step.next()))
  q_to_kv_maps = [q_to_key_value_pairs(qa_pairs, inverted_idx, low_idf_words)
                  for qa_pairs in qa_pairs_data]

  print('{}. Vectorizing QA pairs'.format(step.next()))
  v_qa_pairs_data = [vectorized_qa_pairs(FLAGS, qa_pairs, q_to_kv_map,
                                         vocab, max_q_len)
                     for qa_pairs, q_to_kv_map in zip(qa_pairs_data,
                                                      q_to_kv_maps)]

  data_dict = {}
  data_dict['q_to_kv_maps'] = q_to_kv_maps
  data_dict['v_qa_pairs_data'] = v_qa_pairs_data
  data_dict['vectorized_windows'] = v_wins
  data_dict['vocab'] = vocab
  data_dict['ans_vocab'] = ans_vocab
  data_dict['rev_vocab'] = rev_vocab
  data_dict['rev_ans_vocab'] = rev_ans_vocab
  data_dict['max_sentence_len'] = max_sentence_len
  data_dict['max_q_len'] = max_q_len
  data_dict['doc_to_film_map'] = doc_to_film_map
  with open(processed_data_file, 'wb') as pkl_file:
      pickle.dump(data_dict, pkl_file)
  return data_dict


# This is just for testing data_utils.
if __name__ == "__main__":
  class FLAGS():
    pass
  FLAGS.data_dir = './data'
  FLAGS.input_dir = './movieqa'
  FLAGS.train_dir = './train'
  FLAGS.batch_size = 50
  FLAGS.mem_size = 1000
  FLAGS.pad_word = u'_NIL'
  FLAGS.max_ans_count = 4
  FLAGS.win_file = 'wiki-w=0-d=3-i-m.txt'
  FLAGS.entities_file = 'knowledge_source/entities.txt'
  FLAGS.docs_count = 0
  FLAGS.data_pkl_file = 'processed_data.pkl'
  t = time.time()
  data_dict = prepare_data(FLAGS)
  print time.time() - t
