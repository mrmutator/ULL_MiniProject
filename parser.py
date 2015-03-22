#coding: utf8
import cdec
import numpy as np
import subprocess
import re


def create_cdec_grammar(root_counts, tree_counts):
    terminals = ['0','1','2','3','4','5','6','7','8','9']
    grammar = ""
    for tree in tree_counts:
        root = tree.split()[0]

        logprob = np.log(float(tree_counts[tree])/root_counts[root])

        leaves = re.findall(r" (\w+?)\)", tree)
        i = 0
        for j, l in enumerate(leaves):
            if l not in terminals:
                i += 1
                leaves[j] = "[" + l + "," + str(i) + "]"

        RH = " ".join(leaves)

        rule = "[" + root + "]" + " ||| " + RH + " ||| " + RH + " ||| LogProb=" + str(logprob) + "\n"

        grammar += rule

    return grammar




class Parser(object):

    def __init__(self, config_file, path_cdec):
        self.config_file = config_file
        self.path_cdec = path_cdec


    def get_inside_string(self, string):
        parsing = subprocess.Popen([self.path_cdec, "-c", self.config_file, "-z" ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = parsing.communicate(input=string + "\n")[0]
        return float(re.search(r"log\(Z\): (-?\d+?\.\d+?)[^\d]", result).group(1))

    def get_best_parse(self, string):
        parsing = subprocess.Popen([self.path_cdec, "-c", self.config_file, "-z" ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = parsing.communicate(input=string + "\n")[0]
        return re.search(r"tree: (\(.+\))\n", result).group(1)[1:-1]

    def get_random_parse(self, string):
        # TO-DO: implement
        return None



if __name__ == "__main__":

    grammar_f = open("initial_grammar", "r")
    grammar = grammar_f.read()
    grammar_f.close()

    parser = Parser("initial.ini", "/home/rwechsler/PycharmProjects/cdec/decoder/cdec")


    test = "1 2"


    print parser.get_inside_string(test)
    print parser.get_best_parse(test)


    #tree_counts = {"S (NZ 3) (S2 S2)": 1}
    #root_counts = {"S": 1}

    #print create_cdec_grammar(root_counts, tree_counts)