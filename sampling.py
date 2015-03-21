from __future__ import division
import random
import numpy as np
import re
import sys
import cPickle as pickle
from matplotlib import pyplot as plt
import logging


#TODO: Check if we are gonna use this global dicts
treeFrequency = dict()
treeDerivation = dict()
rootFrequency = dict()
probabilities = dict()
parses = list()


newTreeFrequency = dict()
newRootFrequency = dict()
newProbabilities = dict()

def updateDictionary(parse, update=True,statistcs=True):
    global treeFrequency
    global rootFrequency
    global probabilities
    global newTreeFrequency
    global newRootFrequency
    global newProbabilities
    
    # remove head and tail parenthesis
    parse = parse[1:len(parse)-1]
    parse = parse.strip()
    
    if update:
        # get root
        root = parse.split()[0]
        
        # get subtree
        tree = ' '.join(parse.split()[1:])
        
        if statistcs: # update general stats
            # update frequency of this tree
            treeFrequency.setdefault(parse,0)
            treeFrequency[parse] += 1
            
            # update the root frequency
            rootFrequency.setdefault(root,0)
            rootFrequency[root] += 1
            
            # update tree derivation
            treeDerivation.setdefault(root,set())
            treeDerivation[root].add(tree)
        else: # update new stats
            
            newTreeFrequency.setdefault(parse,0)
            newTreeFrequency[parse] += 1
            
            newRootFrequency.setdefault(root,0)
            newRootFrequency[root] += 1
        
    derivations = set()
    derivations.add(parse)
        
    return derivations


def computeLikelihoodOfParse(parse, probabilities):
    eTrees = decomposeTSG(parse, update=False, statistcs=False)
    probability = 1
    for tree in eTrees:
        #TODO: Call the cdec python wrapper to compute the inside probability of the tree
        continue
    
    return probability


def get_dataset_likelihood(rawBlock, newBlock):
    newLikelihood = 0
    oldLikelihood = 0
    for parse in parses:
        newParse = parse.replace(rawBlock,newBlock)
#             newLikelihood *= computeLikelihoodOfParse(newParse, newProbabilities)
#             oldLikelihood *= computeLikelihoodOfParse(parse, probabilities)
        newLikelihood += computeLikelihoodOfParse(newParse, newProbabilities)
        oldLikelihood += computeLikelihoodOfParse(parse, probabilities)
    
    return oldLikelihood, newLikelihood



# def make_random_candidate_change(dataset):
#
#     # TO-DO: change this so it works for all grammars
#     # Lautaro
#
#     new_dataset = []
#     new_tsg = TSGData()
#
#
#     # get random derivation
#     temp = [d for d in dataset]
#     random.shuffle(temp)
#     for t in temp:
#         num_sites = sum([1 for ch in t if ch == "P"])
#
#         if num_sites > 0:
#             break
#     else:
#         # adding not possible:
#         raise Exception("Adding not possible")
#
#     j = random.randint(0, num_sites-1)
#
#     old_string = t
#
#     new_string = ""
#     s_count = -1
#     for ch in old_string:
#         new_string += ch
#         if ch in ["P"]:
#             s_count += 1
#             if s_count== j:
#                 new_string += "*"
#
#     new_string = new_string.replace("**", "")
#
#     new_trees = get_elementary_trees(new_string)
#     old_trees = get_elementary_trees(old_string)
#
#     new_trees_temp = list(new_trees)
#     removed_trees = []
#     for t in old_trees:
#         try:
#             i = new_trees_temp.index(t)
#             new_trees_temp.pop(i)
#         except ValueError:
#             removed_trees.append(t)
#
#     old_trees_temp = list(old_trees)
#     added_trees = []
#     for t in new_trees:
#         try:
#             i = old_trees_temp.index(t)
#             old_trees_temp.pop(i)
#         except ValueError:
#             added_trees.append(t)
#
#
#     change_type = None
#
#     if len(removed_trees) == 2: # removal
#         old_pattern = removed_trees[0].rstrip(")")[:-1] + removed_trees[1][:2] + "*" + removed_trees[1][2:] + ")"
#         new_pattern = removed_trees[0].rstrip(")")[:-1] + removed_trees[1] + ")"
#         old_pattern = re.sub("P\\)+", "(P", old_pattern)
#         new_pattern = re.sub("P\\)+", "(P", new_pattern)
#         change_type= "removal"
#     elif len(removed_trees) == 1: # add
#         old_pattern = removed_trees[0]
#         new_pattern = added_trees[0].rstrip(")")[:-1] + added_trees[1][:2] + "*" + added_trees[1][2:] + ")"
#         old_pattern = re.sub("P\\)+", "(P", old_pattern)
#         new_pattern = re.sub("P\\)+", "(P", new_pattern)
#         change_type = "add"
#     else:
#         raise Exception("Should not happen.")
#
#     changes = 0
#
#     for d in dataset:
#         new_d = d.replace(old_pattern, new_pattern)
#         if new_d == d:
#             new_d = d.replace(re.sub("^\\(P", "(P*", old_pattern).replace("**", ""), new_pattern)
#
#         if new_d != d:
#             changes += 1
#         new_tsg.add_el_trees(get_elementary_trees(new_d))
#         new_dataset.append(new_d)
#
#     if changes == 0:
#         print change_type
#
#     return new_tsg, new_dataset

def getNonTerminals():
    '''
    Returns a list of non terminal symbols. Does not include the root symbol.
    '''
    
    return ['S1','S2','NZ','Z'] #TODO: Fix this. Get the list dynamically.


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
                match = re.search(r"'.*(?:"+'|'.join(getNonTerminals()) +")'", tag) # match any non terminal symbol but the root
                if match:
                    newTag = match.group()+'*'
                else:
                    newTag = tag
            else:
                newTag = tag
                convertedTree.append(newTag)
        
        
        convertedTreebank.append(convertedTree)

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


def decomposeTSG(tree, update=False, statistcs=False):
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

        if not option: # i got stuck in the while loop
            logging.info('Couldnt find a replacement slot')
            continue

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

        if not option: # i got stuck in the while loop
            logging.info('Couldnt find a replacement slot')
            continue

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
    
    return newParses, rawBlock, newBlock


def metropolis_hastings(old_dataset, n=1000, ap=None, outfile=sys.stdout):
#     old_likelihood = get_dataset_likelihood(old_dataset)

    #TODO: move this write. We dont have the old_likelihood at this point.
    outfile.write("\t".join(["0", "A", str(old_likelihood), str(old_likelihood), str(old_tsg.get_grammar_size()), str(old_tsg.total_trees)]) + "\n")
    
    for i in range(n):
#         new_dataset = make_random_candidate_change(old_dataset) # Lau: new method should return dataset with candidate changes
        new_dataset, rawBlock, newBlock = make_random_candidate_change(old_dataset)
        old_likelihood, new_likelihood = get_dataset_likelihood(rawBlock, newBlock) # lqrz: by passing the old and new block we can forloop only once to ge the likelihood.
        #if new_dataset == old_dataset:
        #    print "EQUAL!!"

        #print new_tsg.get_grammar_size()
        #print new_tsg.total_trees

        if new_likelihood > old_likelihood:
            outfile.write("\t".join([str(i+1), "A", str(new_likelihood), str(new_likelihood), str(new_tsg.get_grammar_size()), str(new_tsg.total_trees)]) + "\n")
            #print "accepted: ", new_likelihood, old_likelihood
            old_likelihood = new_likelihood
            old_dataset = new_dataset
        else:
            if not ap:
                p = np.exp(new_likelihood- old_likelihood)
            else:
                p = ap
            r =np.random.binomial(1, p)
            if r:
                outfile.write("\t".join([str(i+1), "F", str(new_likelihood), str(new_likelihood), str(new_tsg.get_grammar_size()), str(new_tsg.total_trees)]) + "\n")
                #print "forced: ", new_likelihood, old_likelihood
                old_likelihood = new_likelihood
                old_dataset = new_dataset
            else:
                # reject
                outfile.write("\t".join([str(i+1), "R", str(new_likelihood), str(old_likelihood), str(new_tsg.get_grammar_size()), str(new_tsg.total_trees)]) + "\n")
                #print "rejected ", new_likelihood, old_likelihood

        print i, old_likelihood

    return old_dataset

def run_experiment(outfile_name, subset_size=10000, ap=None, iterations=10000):
    # take a subset of numerals from the empirical data

    #num_dist, _ = get_empirical_data("data/wsj01-21-without-tags-traces-punctuation-m40.txt")

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

    subset = get_random_subset(num_dist, size=10000)

    # Assign a parse and randomly mark substitution sites

    # dataset = [random_mark_subst_site(det_parse_num(str(i))) for i in subset] # comment - lqrz

	dataset = place_substitution_points(subset) # lqrz

    outfile = open(outfile_name + "_results.txt", "w")

    final_dataset = metropolis_hastings(dataset, n=iterations, ap=ap, outfile=outfile)

    dmp = [final_tsg, final_dataset]

    pickle.dump(dmp, open(outfile_name+"_grammar.pkl", "wb"))

    outfile.close()

    rules = final_tsg.get_rule_dict()

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



