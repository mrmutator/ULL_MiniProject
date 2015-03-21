__author__ = 'nguyen'

import numpy
import matplotlib.pyplot as plt

class CorpusReader:
    def __init__(self):
        self.lookup = dict()
        self.lookup['one'] = 1
        self.lookup['two'] = 2
        self.lookup['three'] = 3
        self.lookup['four'] = 4
        self.lookup['five'] = 5
        self.lookup['six'] = 6
        self.lookup['seven'] = 7
        self.lookup['eight'] = 8
        self.lookup['nine'] = 9
        self.lookup['ten'] = 10
        self.lookup['eleven'] = 11
        self.lookup['twelve'] = 12
        self.lookup['thirteen'] = 13
        self.lookup['fourteen'] = 14
        self.lookup['fifteen'] = 15
        self.lookup['sixteen'] = 16
        self.lookup['seventeen'] = 17
        self.lookup['eighteen'] = 18
        self.lookup['nineteen'] = 19
        self.lookup['twenty'] = 20
        self.lookup['thirty'] = 30
        self.lookup['forty'] = 40
        self.lookup['fifty'] = 50
        self.lookup['sixty'] = 60
        self.lookup['seventy'] = 70
        self.lookup['eighty'] = 80
        self.lookup['ninety'] = 90
        self.lookup['hundred'] = 100
        self.lookup['thousand'] = 1000
        self.lookup['million'] = 10 ** 6
        self.lookup['billion'] = 10 ** 9
        self.lookup['trillion'] = 10 ** 12
        self.count_num = dict()
        self.count_alpha = dict()
        self.count_total = dict()
        pass


    def read_data(self, filename, tag='CD'):
        '''
        Read number data from a corpus or treebank
        :param filename: Name of file to be read
        :param tag: Look for numbers inside this tag, if None is supplied,
                    numbers are assumed to be one per line
        '''
        with open(filename) as fileobject:
            for line in fileobject:
                number_groups = []

                # If a tag is provided, read their values first
                if tag is not None:
                    extendedtag = tag + ' "'
                    current_number = ''
                    next_tag_idx = line.find(extendedtag)
                    while next_tag_idx != -1:
                        line = line[next_tag_idx+len(tag)+2:]
                        next_quote_idx = line.find('"')
                        if current_number:
                            current_number += ' '
                        current_number += line[:next_quote_idx]

                        next_tag_idx = line.find(extendedtag)
                        if next_tag_idx > len(current_number) + 4:
                            # The two tags are too far away
                            number_groups.append(current_number)
                            current_number = ''
                    if current_number:
                        number_groups.append(current_number)

                    i = 0
                    while i < len(number_groups):
                        # Post-process
                        if number_groups[i].find('/') != -1:
                            number_groups.remove(number_groups[i])
                        else:
                            number_groups[i] = number_groups[i].replace(',', '')
                            number_groups[i] = number_groups[i].replace('-', ' ')
                            i += 1
                else:
                    number_groups.append(line)

                for numberstr in number_groups:
                    numberstr = numberstr.lower()
                    numberstr = numberstr.strip()
                    numbers = numberstr.split(' ')
                    if len(numbers) == 1:
                        numbers = numbers[0]
                        if numbers in self.lookup:
                            number = self.lookup[numbers]
                            self.count_alpha = self.add(self.count_alpha, number)
                            self.count_total = self.add(self.count_total, number)
                        else:
                            try:
                                number = int(numbers)
                                self.count_num = self.add(self.count_num, number)
                                self.count_total = self.add(self.count_total, number)
                            except:
                                continue
                    elif len(numbers) == 2:
                        number = 0
                        isfloat = [False, False]
                        firstnum = 0
                        secondnum = 0

                        try:
                            firstnum = float(numbers[0])
                            isfloat[0] = True
                        except Exception:
                            pass

                        try:
                            secondnum = float(numbers[1])
                            isfloat[1] = True
                        except Exception:
                            pass

                        if isfloat[0] and isfloat[1]:
                            self.count_total = self.add(self.count_total, int(firstnum))
                            self.count_total = self.add(self.count_total, int(secondnum))
                            self.count_num = self.add(self.count_num, int(firstnum))
                            self.count_num = self.add(self.count_num, int(secondnum))
                        else:
                            if isfloat[0] and not isfloat[1]:
                                if numbers[1] in self.lookup:
                                    number = int(firstnum * self.lookup[numbers[1]])
                                else:
                                    number = int(firstnum)
                                self.count_total = self.add(self.count_total, number)
                                self.count_alpha = self.add(self.count_alpha, number)
                            elif not isfloat[0] and not isfloat[1]:
                                adding = True
                                if numbers[0] not in self.lookup or numbers[1] not in self.lookup:
                                    if numbers[1][-1] == 's':
                                        try:
                                            number = int(numbers[1][:-1])
                                        except Exception:
                                            adding = False
                                else:
                                    number = self.lookup[numbers[0]] * self.lookup[numbers[1]]
                                if adding:
                                    self.count_total = self.add(self.count_total, number)
                                    self.count_alpha = self.add(self.count_alpha, number)


    def get_statistics(self, limit=3000):
        keys = range(limit)
        number_form = []
        alphabetic_form = []
        total = []
        for number in keys:
            if number in self.count_num:
                number_form.append(self.count_num[number])
            else:
                number_form.append(0)

            if number in self.count_alpha:
                alphabetic_form.append(self.count_alpha[number])
            else:
                alphabetic_form.append(0)

            if number in self.count_total:
                total.append(self.count_total[number])
            else:
                total.append(0)
        return number_form, alphabetic_form, total


    def getIndexFromProb(self, probList, randomValue):
        probArray = numpy.array(probList)
        probArray = probArray * 1. / numpy.sum(probArray, axis=0)
        cumprob = numpy.cumsum(probArray)
        return numpy.size(cumprob, 0) - numpy.count_nonzero(cumprob > randomValue)


    def sample(self, limit=4000, size=20000, uniformprob=0.001):
        samples = numpy.zeros(limit)
        _, _, total = self.get_statistics(limit)
        for _ in range(size):
            if numpy.random.sample() < uniformprob:
                # Sample uniformly
                samples[numpy.random.randint(0, limit)] += 1
            else:
                samples[self.getIndexFromProb(total, numpy.random.sample())] += 1
        return samples

    def add(self, dictToAdd, key):
        if key not in dictToAdd:
            dictToAdd[key] = 0
        dictToAdd[key] += 1

        return dictToAdd


if __name__ == '__main__':
    # Initialize the reader
    reader = CorpusReader()

    # Read numeral data
    reader.read_data('wsj01-21-without-tags-traces-punctuation-m40.txt', 'CD')
#     reader.read_data('numbers', None)

    # Limit to range [0,200)
    limit = 200

    # Get distributions
    number_form, alphabetic_form, total = reader.get_statistics(limit=limit)

    keys = range(limit)

    # Plot the true distribution
    plt.bar(keys, alphabetic_form, color='g')
    plt.show()

    # Sample from the distribution
    samples = reader.sample(limit, size=20000, uniformprob=0.001)

    # Plot the sampled distribution
    plt.bar(keys, samples, color='g')
    plt.show()
    pass