from __future__ import division
import random
import numpy as np
import re
import sys
import cPickle as pickle
from matplotlib import pyplot as plt
from datareader import CorpusReader
from parser import Parser, create_cdec_grammar
import os
import constants


#----------- Begin Params ---------#
CDEC_PATH = constants.CDEC_PATH
WEIGHT_FILE = constants.WEIGHT_FILE
INITIAL_INI = constants.INITIAL_INI
TMP_DATA_DIR = constants.TMP_DATA_DIR
#----------- End Params -----------#


#----------- Begin global vars ---------#
treeFrequency = dict() # Keys: Elementary trees. Values: Frequency of each elementary tree
rootFrequency = dict() # Keys: Roots of the elementary trees. Values: Frequency of each root
newTreeFrequency = dict() # Keys: Elementary trees after Metrop-Hast random change. Values: Frequency of each elementary tree
newRootFrequency = dict() # Keys: Roots of the elementary trees formed after Metrop-Hast random change. Values: Frequency of each elementary tree
#----------- End global vars -----------#


def updateDictionary(parse, update=True,statistcs=True):
    '''
    Updates the frequency dictionaries
    '''
    
    global treeFrequency
    global rootFrequency
    global newTreeFrequency
    global newRootFrequency
    
    # remove head and tail parenthesis
    parse = parse[1:len(parse)-1]
    parse = parse.strip()
    
    if update:
        # get root
        root = parse.split()[0]
        
        if statistcs: # update general stats
            # update frequency of this tree
            treeFrequency.setdefault(parse,0)
            treeFrequency[parse] += 1
            
            # update the root frequency
            rootFrequency.setdefault(root,0)
            rootFrequency[root] += 1

        else: # update new stats
            
            newTreeFrequency.setdefault(parse,0)
            newTreeFrequency[parse] += 1
            
            newRootFrequency.setdefault(root,0)
            newRootFrequency[root] += 1
        
    derivations = set()
    derivations.add(parse)
        
    return derivations


def parse_dataset(dataset, ini_file):

    p = parser.Parser(ini_file, CDEC_PATH)
    parsed = []
    for string in dataset:
        parse = p.get_best_parse(' '.join(string))
        parsed.append(parse)

    return parsed



def get_dataset_likelihood(raw_dataset, root_counts, tree_counts):
    '''
    Computes the likelihood of the whole dataset
    '''

    # generate grammar
    grammar = create_cdec_grammar(root_counts, tree_counts)

    # write grammar_file

    outfile = open(TMP_DATA_DIR + "tmp_grammar.cfg", "w")
    outfile.write(grammar)
    outfile.close()

    # write init_file

    infile = open(INITIAL_INI, "r")
    init = infile.read()

    infile.close()

    new_init = re.sub("grammar=.*?", "grammar=tmp_grammar.cfg")

    outfile = open(TMP_DATA_DIR + "tmp_init.ini", "w")
    outfile.write(new_init)
    outfile.close()

    parser = Parser(TMP_DATA_DIR + "tmp_init.ini", CDEC_PATH)

    likelihood = 0

    for s in dataset:
        likelihood += parser.get_inside_string(" ".join(s))

    # delete tmp_files
    os.remove(TMP_DATA_DIR + "tmp_grammar.cfg")
    os.remove(TMP_DATA_DIR + "tmp_init.ini")

    return likelihood

def getNonTerminals():
    '''
    Returns a list of non terminal symbols. Does not include the root symbol.
    '''

    return ['S1', 'S2', 'D', 'NZ'] #TODO: Fix this. Get the list dynamically.


def placeSubstitutionPoints(treebank):
    '''
    Receives a list of trees and randomly places subtitution points.
    :param treebank: Treebank corpus
    :return: Marked treebank corpus
    '''

    convertedTreebank = []

    threshold = 0.3

    for tree in treebank:
        convertedTree = []

        tags = tree.split(' ')
        print tags
        for tag in tags:
            print tag
#             match = re.match(r'.*S ',n).group()
#             if tag=='(S':

            p = np.random.rand()

            if p < threshold:
                match = re.search(r'.*(?:'+'|'.join(getNonTerminals()) +')', tag) # match any non terminal symbol but the root
                if match:
                    newTag = match.group()+'*'
                else:
                    newTag = tag
            else:
                newTag = tag

            convertedTree.append(newTag)


        convertedTreebank.append(' '.join(convertedTree))

    return convertedTreebank


def getBlock(firstStarIndex, tree):
    '''
    Retrieves the elementary tree of a marked tee
    :param firstStarIndex: Index of the '*' symbol in the tree
    :param tree: tree containing the elementary tree to be extracted
	:return:
	'''
    # Get first previous parenthesis
    for i in range(firstStarIndex):
        firstParenthesis = tree.find('(',firstStarIndex-i,firstStarIndex)
        if firstParenthesis > -1:
            break

    # get block by counting parenthesis
    count = 0
    for i,ch in enumerate(list(tree[firstParenthesis:])):
        if ch=='(':
            count += 1
        if ch==')':
            count -= 1

        if count == 0:
            break

    lastParenthesis = firstParenthesis+i+1 # include the last parenthesis

    # get block
    rawBlock = tree[firstParenthesis:lastParenthesis]

    # get substitution symbol
    substitutionSymbol = tree[firstParenthesis+1:firstStarIndex]

    return rawBlock, substitutionSymbol


def decomposeTSG(tree,update=False,statistcs=False):
    '''
    Decomposes a tree which has substitution points into elementary trees.
    :param tree: marked tree with substitution points
    :return:
    '''

    if tree.count('*')==0:
        derivations = updateDictionary(tree,update=update,statistcs=statistcs)
        
        return derivations

    # Get first star
    firstStarIndex = tree.find('*') # returns -1 on failure

    rawBlock, substitutionSymbol = getBlock(firstStarIndex, tree)

    # remove first star from block
    block = rawBlock.replace('*','',1) # theres always gonna be, at least, one.

    # replace the block by the substitution symbol
    newL = tree.replace(rawBlock, ' ('+substitutionSymbol+') ', 1)

    result = set()
    # recursive call to keep decomposing
    result = result.union(decomposeTSG(newL,update=update,statistcs=statistcs))
    result = result.union(decomposeTSG(block,update=update,statistcs=statistcs))

    return result


def make_random_candidate_change(treebank):
    '''
    Given a grammar, makes a random change (adds or removes substitution point) and
	retrieves the new elementary trees.
	:param dataset:
	:return:
	'''

    pChange = random.random() # add or delete star?

    option = False
    prevent = 0 # prevent the candidate search while loops from getting stuck
    probAdding = 0.5 # probability of making a change by adding a star

    if pChange < probAdding: # add a star
        # select a parse that has a slot to add a star
        while not option:

            if prevent>len(treebank):
                break

            parseIndex = random.randint(0, len(treebank)-1) # select a random parse
            parse = treebank[parseIndex]
            symbols = getNonTerminals()
            slots = [symbol for symbol in symbols if parse.count('('+symbol+' ')>0 or parse.count(' '+symbol+' ')>0]
            if len(slots) > 0:
                option = True

            prevent += 1

        #TODO: Check this. Before, I would move to the next iteration
#         if not option: # i got stuck in the while loop
#             logging.info('Couldnt find a replacement slot')
#             continue

        symbol = slots[random.randint(0,len(slots)-1)] # choose random symbol to insert star
        countSymbol = parse.count(symbol+' ') # take care not to count symbol*
        pSymbol = random.randint(1,countSymbol)

        # get index of chosen symbol
        index=0
        for _ in range(pSymbol):
            index = parse.find(symbol+' ', index+1,len(parse)) # symbols have variable length

        # get block and substitution symbol
        rawBlock, _ = getBlock(index, parse)

        # remove first star from block
        newBlock = rawBlock.replace(symbol,symbol+'*',1) # theres always gonna be, at least, one.

    else: # delete a star
        # select a parse that has a star to be removed
        while not option:

            if prevent>len(treebank):
                break

            parseIndex = random.randint(0, len(treebank)-1) # select a random parse
            parse = treebank[parseIndex]
            if parse.count('*') > 0:
                option = True

            prevent += 1

        #TODO: Check this. Before, I would move to the next iteration
#         if not option: # i got stuck in the while loop
#             logging.info('Couldnt find a replacement slot')
#             continue

        # remove a star
        countStars = parse.count('*')
        pStar = random.randint(1,countStars) # choose star to eliminate

        # get index of chosen star
        index=0
        for _ in range(pStar):
            index = parse.find('*', index+1,len(parse))

        # get block and substitution symbol
        rawBlock, _ = getBlock(index, parse)

        # remove first star from block
        newBlock = rawBlock.replace('*','',1) # theres always gonna be, at least, one.

    newParses = list()
    for tree in treebank:
        newParse = tree.replace(rawBlock,newBlock)
        newParses.append(newParse)
        decomposeTSG(newParse, update=True,statistcs=False) # stats==True: update general

    return newParses


def metropolis_hastings(raw_dataset, old_dataset, n=1000, ap=None, outfile=sys.stdout):
    '''
    Runs Metropolis Hastings algorithm
    '''
    
    global treeFrequency
    global rootFrequency
    
    old_likelihood = get_dataset_likelihood(old_dataset)

    outfile.write("\t".join(["0", "A", str(old_likelihood), str(old_likelihood), str(len(treeFrequency.keys())), str(np.sum(treeFrequency.values()))]) + "\n")

    for i in range(n):

        new_dataset, new_tsg, _, _ = make_random_candidate_change(old_dataset)
        new_likelihood = get_dataset_likelihood(raw_dataset, rootFrequency, treeFrequency) # lqrz: by passing the old and new block we can forloop only once to ge the likelihood.
        #if new_dataset == old_dataset:
        #    print "EQUAL!!"

        #print new_tsg.get_grammar_size()
        #print new_tsg.total_trees

        if new_likelihood > old_likelihood:
            outfile.write("\t".join([str(i+1), "A", str(new_likelihood), str(new_likelihood), str(len(newTreeFrequency.keys())), str(np.sum(newTreeFrequency.values()))]) + "\n")
            #print "accepted: ", new_likelihood, old_likelihood
            old_likelihood = new_likelihood
            old_dataset = new_dataset
            treeFrequency = newTreeFrequency
            rootFrequency = newRootFrequency
        else:
            if not ap:
                p = np.exp(new_likelihood- old_likelihood)
            else:
                p = ap
            r =np.random.binomial(1, p)
            if r:
                outfile.write("\t".join([str(i+1), "F", str(new_likelihood), str(new_likelihood), str(len(newTreeFrequency.keys())), str(np.sum(newTreeFrequency.values()))]) + "\n")
                #print "forced: ", new_likelihood, old_likelihood
                old_likelihood = new_likelihood
                old_dataset = new_dataset
                treeFrequency = newTreeFrequency
                rootFrequency = newRootFrequency
            else:
                # reject
                outfile.write("\t".join([str(i+1), "R", str(new_likelihood), str(old_likelihood), str(len(newTreeFrequency.keys())), str(np.sum(newTreeFrequency.values()))]) + "\n")
                #print "rejected ", new_likelihood, old_likelihood

        print i, old_likelihood

    return old_dataset

def run_experiment(outfile_name, subset_size=10000, ap=None, iterations=10000):
    
    global treeFrequency
    global rootFrequency
    
    # take a subset of numerals from the empirical data

    #num_dist, _ = get_empirical_data("data/wsj01-21-without-tags-traces-punctuation-m40.txt")

    reader = CorpusReader()

#     data = reader.read_data('numbers', None)
    reader.read_data('wsj01-21-without-tags-traces-punctuation-m40.txt', 'CD')

    grammar_f = open("example.cfg", "r")
    grammar = grammar_f.read()
    grammar_f.close()

    parser = Parser(grammar, "example.weights")

    parses = []
    for s in data:
        parses.append(parser.get_best_parse(s))

    x, y = zip(*[(x,num_dist[x]) for x in num_dist.keys() if x <= 100])

    plt.figure()
    plt.bar(x,y)
    plt.savefig(outfile_name + "init" + ".png")
    numbers = []
    for n in num_dist.keys():
        if n < 1000:
            numbers += [n] * num_dist[n]

    plt.figure()
    plt.hist(numbers, bins=10)
    plt.savefig(outfile_name + "init_b" + ".png")


    # Limit to range [0,4000)
    # With really small probability, perform uniform sampling
    # The return variable is a numpy 1D array
    subset = reader.sample(limit=4000, size=20000, uniformprob=0.001)

    # Assign a parse and randomly mark substitution sites

    # dataset = [random_mark_subst_site(det_parse_num(str(i))) for i in subset] # comment - lqrz

    dataset = placeSubstitutionPoints(subset) # lqrz

    outfile = open(outfile_name + "_results.txt", "w")

    final_dataset = metropolis_hastings(dataset, n=iterations, ap=ap, outfile=outfile)

#     dmp = [final_tsg, final_dataset]
    dmp = [treeFrequency, rootFrequency, final_dataset]

    pickle.dump(dmp, open(outfile_name+"_grammar.pkl", "wb"))

    outfile.close()

#     rules = final_tsg.get_rule_dict()
    rules = treeFrequency.keys() # the keys are gonna have all the elementary trees

    cum_rules = transfer_rules(rules)

    terminals = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # only base dist

    freq_dict = sample_rules(cum_rules, terminals, n=10000)

    x, y = zip(*[(x,freq_dict[x]) for x in freq_dict.keys() if x <= 100])

    plt.figure()
    plt.bar(x,y)
    plt.savefig(outfile_name + "final" + ".png")

    numbers = []
    for n in freq_dict.keys():
        if n < 1000:
            numbers += [n] * num_dist[n]

    plt.figure()
    plt.hist(numbers, bins=10)
    plt.savefig(outfile_name + "final_b" + ".png")



    print "Experiment " + outfile_name + " done."


#run_experiment("results/10000_2000", subset_size=10000, ap=None, iterations=2000)
#run_experiment("results/10000_2000_001", subset_size=10000, ap=0.01, iterations=2000)


#---------------- For debugging purposes

reader = CorpusReader()

reader.read_data('numbers', None)
data = reader.count_total



parser = Parser(INITIAL_INI, CDEC_PATH)

parses = []
raw_dataset = []
for s in data:
    raw_dataset.append(s)
    s = ' '.join(str(s))
    parses.append(parser.get_best_parse(s))

#parses = ['S (D 1)','S (S1 (NZ 8)) (S2 (D 0) (S2 (D 0) (S2 (D 0))))', 'S (S1 (NZ 3)) (S2 (D 5))']

dataset = placeSubstitutionPoints(parses)
final_dataset = metropolis_hastings(raw_dataset, dataset, n=10)
