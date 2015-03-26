__author__ = 'rwechsler'
import constants
from parser import Parser
import re
import cPickle as pickle
import os
from datareader import CorpusReader

#----------- Begin Params ---------#
CDEC_PATH = constants.CDEC_PATH
WEIGHT_FILE = constants.WEIGHT_FILE
INITIAL_INI = constants.INITIAL_INI
TMP_DATA_DIR = constants.TMP_DATA_DIR
#----------- End Params -----------#



def get_dataset_likelihood(raw_dataset, grammar):
    '''
    Computes the likelihood of the whole dataset
    '''

    # write grammar_file

    outfile = open(TMP_DATA_DIR + "tmp_grammar.cfg", "w")
    outfile.write(grammar)
    outfile.close()

    # write init_file

    infile = open(INITIAL_INI, "r")
    init = infile.read()

    infile.close()

    new_init = re.sub("grammar=.*", "grammar=" + TMP_DATA_DIR + "tmp_grammar.cfg", init)

    outfile = open(TMP_DATA_DIR + "tmp_init.ini", "w")
    outfile.write(new_init)
    outfile.close()

    parser = Parser(TMP_DATA_DIR + "tmp_init.ini", CDEC_PATH)

    likelihood = 0

    prob_f = parser.get_inside_string

    for s in raw_dataset:
        likelihood += prob_f(" ".join(s))

    # delete tmp_files
    os.remove(TMP_DATA_DIR + "tmp_grammar.cfg")
    os.remove(TMP_DATA_DIR + "tmp_init.ini")

    return likelihood


_, _, _, final_grammar1 = pickle.load(open('results/comp_vit_final.pkl', 'rb'))
_, _, _, final_grammar2 = pickle.load(open('results/comp_noinside_final.pkl', 'rb'))


reader = CorpusReader()
reader.read_data('wsj01-21-without-tags-traces-punctuation-m40.txt', 'CD')

dist = reader.sample(limit=4000, size=1000)
raw_dataset = []
for i, n in enumerate(dist):
    raw_dataset += [str(i)] * n


print get_dataset_likelihood(raw_dataset, final_grammar2)
print get_dataset_likelihood(raw_dataset, final_grammar1)