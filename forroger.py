'''
Created on Mar 22, 2015

@author: nguyen
'''

import cPickle as pickle
import numpy
import matplotlib.pyplot as plt

class Tree:
    def __init__(self, root, left=None, right=None):
        self.root = root
        self.left = left
        self.right = right


    def terminal(self):
        return self.isdigit(self.root)


    def leaf(self):
        return True if self.left is None and self.right is None else False


    def fullygrown(self):
        if self.leaf():
            if self.terminal():
                return True
            else:
                return False
        else:
            return True if self.left.fullygrown() and (self.right is None or self.right.fullygrown()) else False


    def isdigit(self, value):
        try:
            int(value)
            return True
        except Exception:
            return False



class Sampler:
    def __init__(self, final_treeFrequency):
        self.rule_S = []
        self.rule_S1 = []
        self.rule_S2 = []
        self.rule_D = []

        self.rule_S_count = []
        self.rule_S1_count = []
        self.rule_S2_count = []
        self.rule_D_count = []

        for key, value in final_treeFrequency.items():
            if key[0] == '(':
                key = key[1:-1]
            root = key[:2]
            if root == 'S ':
                self.rule_S.append(key)
                self.rule_S_count.append(value)
            elif root == 'S1':
                self.rule_S1.append(key)
                self.rule_S1_count.append(value)
            elif root == 'S2':
                self.rule_S2.append(key)
                self.rule_S2_count.append(value)
            elif root == 'D ':
                self.rule_D.append(key)
                self.rule_D_count.append(value)


    def getleftstr(self, treestr):
        firstleft = treestr.find('(')

        if firstleft == -1:
            return 0, len(treestr)-1

        leftcount = 1
        for i in range(firstleft+1, len(treestr)):
            if treestr[i] == '(':
                leftcount += 1
            elif treestr[i] == ')':
                leftcount -= 1
            if leftcount == 0:
                return firstleft, i


    def parse(self, treestr):
        if not treestr:
            return None
        treestr = treestr.strip()
        if treestr[0] == '(':
            treestr = treestr[1:-1]

        root = treestr[:treestr.find(' ')]
        leftbracket = treestr.find('(')

        if leftbracket == -1:
            spaceidx = treestr.find(' ')
            if spaceidx != -1:
                # Base case, for example 'NZ 4'
                innervalue = treestr[spaceidx+1:]
                return Tree(root, Tree(innervalue))
            else:
                # Base case ending with non-terminal, for example '(S2)'
                return Tree(treestr)

        leftlidx, leftridx = self.getleftstr(treestr)
        leftstr = treestr[leftlidx:leftridx+1]
        rightstr = treestr[leftridx+2:]
        rightstr = rightstr[rightstr.find('('):]

        return Tree(root, self.parse(leftstr), self.parse(rightstr))

    def getnumberstr(self, tree):
        '''
        Get the number a tree represents
        :param tree: A fully-grown tree
        '''
        if tree is None:
            return ''
        if tree.terminal():
            return tree.root
        numberstr = ''
        numberstr += self.getnumberstr(tree.left)
        numberstr += self.getnumberstr(tree.right)
        return numberstr


    def getnumber(self, tree):
        return int(self.getnumberstr(tree))


    def sample(self, limit=4000, size=10000):
        '''
        Sample natural numbers from the grammar
        :param limit: Cut off at this limit
        :param size: Number of samples to return
        '''
        samples = numpy.zeros(limit)
        for _ in range(size):
            number = self.getnumber(self.generate_tree())
            if number < limit:
                samples[number] += 1
        return samples



    def getIndexFromProb(self, probList, randomValue):
        probArray = numpy.array(probList)
        probArray = probArray * 1. / numpy.sum(probArray, axis=0)
        cumprob = numpy.cumsum(probArray)
        return numpy.size(cumprob, 0) - numpy.count_nonzero(cumprob > randomValue)


    def expand_tree(self, tree):
        while not tree.fullygrown():
            if tree.leaf():
                if not tree.terminal():
                    root = tree.root
                    if root == 'S1':
                        treestr = self.rule_S1[self.getIndexFromProb(self.rule_S1_count, numpy.random.sample())]
                    elif root == 'S2':
                        treestr = self.rule_S2[self.getIndexFromProb(self.rule_S2_count, numpy.random.sample())]
                    elif root == 'D':
                        treestr = self.rule_D[self.getIndexFromProb(self.rule_D_count, numpy.random.sample())]
                    subtree = self.parse(treestr)
                    tree.left = subtree.left
                    tree.right = subtree.right
                    return tree
            else:
                if not tree.left.fullygrown():
                    tree.left = self.expand_tree(tree.left)
                if tree.right is not None and not tree.right.fullygrown():
                    tree.right = self.expand_tree(tree.right)
        return tree


    def generate_tree(self):
        treestr = self.rule_S[self.getIndexFromProb(self.rule_S_count, numpy.random.sample())]
        print treestr
        return self.expand_tree(self.parse(treestr))




if __name__ == '__main__':
    inal_dataset, final_rootFrequency, final_treeFrequency, final_grammar = pickle.load(open('deterministic_final.pkl', 'rb'))
#     initial_dist = pickle.load(open('test_initial_dist.pkl', 'rb'))

    # Limit to range [0,200)
    limit = 4000

    keys = range(limit)
    size = 200000


    # Sample from the grammar
    sampler = Sampler(final_treeFrequency)
    samples = sampler.sample(limit=limit, size=size)



    # Plot the sampled distribution
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title(str(size) + ' natural numbers in [0, ' + str(limit) + ') sampled from the deterministic grammar')
    plt.bar(keys, samples, color='g')
    plt.show()

