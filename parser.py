#coding: utf8
import cdec
import gzip
from math import exp



def create_cdec_grammar(root_counts, tree_counts):
    grammar = ""
    for tree in tree_counts:
        r = tree.split()[0]







class Parser(object):

    def __init__(self, grammar, weight_file):
        # Load decoder width configuration
        self.decoder = cdec.Decoder(formalism='scfg')

        # Read weights
        self.decoder.read_weights(weight_file)

        self.grammar = grammar




    def get_inside_string(self, string):
        forest = self.decoder.translate(string, grammar=self.grammar)
        I = forest.inside()
        return I[-1]

    def get_best_parse(self, string):
        forest = self.decoder.translate(string, grammar=self.grammar)
        return forest.viterbi_trees()[0][1:-1]

    def get_random_parse(self, string):
        forest = self.decoder.translate(string, grammar=self.grammar)
        return list(forest.sample_trees(1))[0][1:-1]



if __name__ == "__main__":

    grammar_f = open("example.cfg", "r")
    grammar = grammar_f.read()
    grammar_f.close()

    parser = Parser(grammar, "example.weights")


    print parser.get_inside_string("a")

    print parser.get_best_parse("a")

    print parser.get_random_parse("a")
