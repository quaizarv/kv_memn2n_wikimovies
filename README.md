# Implementation of the "Key-Value Memory Networks for Directly Reading Documents"

Authors: Alexander H. Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, Jason Weston

https://arxiv.org/abs/1606.03126

## Description

Tensorflow implementation for directly reading documents (aka machine
comprehension) on the WikiMovies Corpus. Note that this code is quite different
from the lua/torch implementation from FB.


## Requirements

Download the data corpus from:

http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz

Also we need 2 scripts from the facebook MemNN github repository:

https://github.com/facebook/MemNN

cd to KVmemnn/examples/WikiMovies directory in the FB MemNN  repository

Process wiki docs to generate windows of size 7 as follows:

> ./gen_wiki_windows.sh

The following relies on wikiwindows.py in the KVmemnn/examples/WikiMovies directory

This generates a file named wiki-w=0-d=3-i-m.txt in ./data/ directory

Process questions

> ./gen_multidict_questions.sh

This relies on query_multidict.py file bin the KVmemnn/examples/WikiMovies directory

This generates train_1.txt, test_1.txt and dev_1.txt files in ./data/ directory


## Set the location for the movieqa Corpus and training directory

Update the following parameters in train.py:

Assuming that you have placed the above generated wiki-w=0-d=3-i-m.txt,
train_1.txt, test_1.txt and dev_1.txt files in ./data and the WikiMovies Corpus
in ./movieqa

> FLAGS.input_dir = './movieqa'

> FLAGS.train_dir = './train'

> FLAGS.data_dir = './data'

## Training
Set the mode paramter in train.py to 'train' and then run train.py
> FLAGS.mode = 'train'

> python train.py

Takes about 5 hours to converge on a GPU

## Testing
Set the mode paramter in train.py to 'test' and then run train.py
> FLAGS.mode = 'test'

> python train.py

Is able to replicate the results in the paper