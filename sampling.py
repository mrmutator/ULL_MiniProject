from __future__ import division
import random
import numpy as np
import re
import sys
import cPickle as pickle
from matplotlib import pyplot as plt

def get_dataset_likelihood(dataset):
    pass
    # TO-DO implement



def make_random_candidate_change(dataset):

    # TO-DO: change this so it works for all grammars
    # Lautaro

    new_dataset = []
    new_tsg = TSGData()


    # get random derivation
    temp = [d for d in dataset]
    random.shuffle(temp)
    for t in temp:
        num_sites = sum([1 for ch in t if ch == "P"])

        if num_sites > 0:
            break
    else:
        # adding not possible:
        raise Exception("Adding not possible")

    j = random.randint(0, num_sites-1)

    old_string = t

    new_string = ""
    s_count = -1
    for ch in old_string:
        new_string += ch
        if ch in ["P"]:
            s_count += 1
            if s_count== j:
                new_string += "*"

    new_string = new_string.replace("**", "")

    new_trees = get_elementary_trees(new_string)
    old_trees = get_elementary_trees(old_string)

    new_trees_temp = list(new_trees)
    removed_trees = []
    for t in old_trees:
        try:
            i = new_trees_temp.index(t)
            new_trees_temp.pop(i)
        except ValueError:
            removed_trees.append(t)

    old_trees_temp = list(old_trees)
    added_trees = []
    for t in new_trees:
        try:
            i = old_trees_temp.index(t)
            old_trees_temp.pop(i)
        except ValueError:
            added_trees.append(t)


    change_type = None

    if len(removed_trees) == 2: # removal
        old_pattern = removed_trees[0].rstrip(")")[:-1] + removed_trees[1][:2] + "*" + removed_trees[1][2:] + ")"
        new_pattern = removed_trees[0].rstrip(")")[:-1] + removed_trees[1] + ")"
        old_pattern = re.sub("P\\)+", "(P", old_pattern)
        new_pattern = re.sub("P\\)+", "(P", new_pattern)
        change_type= "removal"
    elif len(removed_trees) == 1: # add
        old_pattern = removed_trees[0]
        new_pattern = added_trees[0].rstrip(")")[:-1] + added_trees[1][:2] + "*" + added_trees[1][2:] + ")"
        old_pattern = re.sub("P\\)+", "(P", old_pattern)
        new_pattern = re.sub("P\\)+", "(P", new_pattern)
        change_type = "add"
    else:
        raise Exception("Should not happen.")

    changes = 0

    for d in dataset:
        new_d = d.replace(old_pattern, new_pattern)
        if new_d == d:
            new_d = d.replace(re.sub("^\\(P", "(P*", old_pattern).replace("**", ""), new_pattern)

        if new_d != d:
            changes += 1
        new_tsg.add_el_trees(get_elementary_trees(new_d))
        new_dataset.append(new_d)

    if changes == 0:
        print change_type

    return new_tsg, new_dataset




def metropolis_hastings(old_dataset, n=1000, ap=None, outfile=sys.stdout):
    old_likelihood = get_dataset_likelihood(old_dataset)


    outfile.write("\t".join(["0", "A", str(old_likelihood), str(old_likelihood), str(old_tsg.get_grammar_size()), str(old_tsg.total_trees)]) + "\n")
    for i in range(n):
        new_dataset = make_random_candidate_change(old_dataset) # Lau: new method should return dataset with candidate changes
        new_likelihood = get_dataset_likelihood(new_dataset) # 
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

    dataset = [random_mark_subst_site(det_parse_num(str(i))) for i in subset]

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



