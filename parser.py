#coding: utf8
import cdec
import gzip
from math import exp

class Parser(object):

    def __init__(self, weight_file="weights"):
        # Load decoder width configuration
        self.decoder = cdec.Decoder(formalism='scfg')

        # Read weights
        self.decoder.read_weights(weight_file)

    def update_grammar(self, grammar):
        self.grammar = grammar


    def get_inside_string(self, string):
        forest = self.decoder.translate(string, grammar=self.grammar)
        I = forest.inside()
        return I[-1]

    def get_best_parse(self, string):
        forest = self.decoder.translate(string, grammar=self.grammar)
        return forest.viterbi()



if __name__ == "__main__":
    parser = Parser("example.weights")

    grammar_f = open("example.cfg", "r")
    grammar = grammar_f.read()
    grammar_f.close()

    parser.update_grammar(grammar)

    print parser.get_inside_string("a")
