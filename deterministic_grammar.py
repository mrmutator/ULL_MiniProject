from bokeh.properties import Size
__author__ = 'nguyen'

import sys
import matplotlib.pyplot as plt
import numpy
import operator
from helper import add

def getIndexFromProb(probList, randomValue):
    cumprob = numpy.cumsum(probList)
    return numpy.size(cumprob, 0) - numpy.count_nonzero(cumprob > randomValue)

def drawRule(rulecount):
    keys = list(rulecount.keys())
    values = numpy.array([rulecount[key] * 1. for key in keys])
    valuesum = numpy.sum(values, axis=0)
    values /= valuesum
    return keys[getIndexFromProb(values, numpy.random.sample())]

def getNumberStats(filename = 'numbers'):
    lookup = dict()
    lookup['one'] = 1
    lookup['two'] = 2
    lookup['three'] = 3
    lookup['four'] = 4
    lookup['five'] = 5
    lookup['six'] = 6
    lookup['seven'] = 7
    lookup['eight'] = 8
    lookup['nine'] = 9
    lookup['ten'] = 10
    lookup['eleven'] = 11
    lookup['twelve'] = 12
    lookup['thirteen'] = 13
    lookup['fourteen'] = 14
    lookup['fifteen'] = 15
    lookup['sixteen'] = 16
    lookup['seventeen'] = 17
    lookup['eighteen'] = 18
    lookup['nineteen'] = 19
    lookup['twenty'] = 20
    lookup['thirty'] = 30
    lookup['forty'] = 40
    lookup['fifty'] = 50
    lookup['sixty'] = 60
    lookup['seventy'] = 70
    lookup['eighty'] = 80
    lookup['ninety'] = 90
    lookup['hundred'] = 100
    lookup['thousand'] = 1000
    lookup['million'] = 10 ** 6
    lookup['billion'] = 10 ** 9
    lookup['trillion'] = 10 ** 12

    count_num = dict()
    count_alpha_n = dict()
    count_alpha_a = dict()
    count_total = dict()

    with open(infile) as fileobject:
            for line in fileobject:
                newline = line.lower()
                newline = newline.replace('  ',' ')
                newline = newline.strip()
                numbers = newline.split(' ')
                if len(numbers) == 1:
                    numbers = numbers[0]
                    if numbers in lookup:
                        number = lookup[numbers]
                        count_alpha_a = add(count_alpha_a, numbers)
                        count_alpha_n = add(count_alpha_n, number)
                        count_total = add(count_total, number)
                    else:
                        try:
                            number = int(numbers)
                            count_num = add(count_num, number)
                            count_total = add(count_total, number)
                        except:
                            continue
                else:
                    count_alpha_a = add(count_alpha_a, newline)
                    try:
                        number = int(float(numbers[0]) * lookup[numbers[1]])
                    except:
                        number = lookup[numbers[0]] + lookup[numbers[1]]
                    count_total = add(count_total, number)
                    count_alpha_n = add(count_alpha_n, number)

    return count_total, count_num, count_alpha_n, count_alpha_a


def limitStats(count_total, limit = 200):
    keylist = range(limit)
    valuelist = []
    for i in keylist:
        if i not in count_total:
            valuelist.append(0)
        else:
            valuelist.append(count_total[i])
    return valuelist


def getParse(number):
    if number < 10:
        return 'S ' + str(number)
    else:
        numstr = str(number)
        parse = 'S ' + numstr[0]
        for i in numpy.arange(1, len(numstr)):
            parse += 'P ' + numstr[i]
        return parse

def numberOfP(number):
    return len(str(number)) - 1

if __name__ == '__main__':
    if len(sys.argv) == 2:
        infile = sys.argv[1]
    else:
        infile = 'numbers'

    count_total, _ , _, _ = getNumberStats(infile)

    # Parameters
    limit = 3000
    n_iterations = 1000
    n_samples = 17000
    #


    valuelist = limitStats(count_total, limit=limit)

    rulecount = dict()
    rulecount['S'] = numpy.sum(valuelist, axis=0)

    total_p = 0

    for i in range(10):
        rulecount['S ' + str(i)] = valuelist[i]

    bank = []

    # For each number 'number'
    for number in numpy.arange(11, limit):
        parse = getParse(number)
        p_count = numberOfP(number)
        p_indices = numpy.arange(len(parse))[3::3]
        # For a single data point i
        for i in range(valuelist[number]):
            trees = []
            random_bin_vec = numpy.random.random_integers(0, 1, size=p_count)
            chosen_p = numpy.array([p_indices[j] for j in range(p_count) if random_bin_vec[j]])
            total_p += len(chosen_p)
            if len(chosen_p):
                trees.append(parse[:chosen_p[0]+1])
                add(rulecount, trees[-1])

                for j in range(len(chosen_p)-1):
                    trees.append(parse[chosen_p[j]:chosen_p[j+1]+1])
                    add(rulecount, trees[-1])

                trees.append(parse[chosen_p[-1]:])
                add(rulecount, trees[-1])

                bank.append(trees)
            else:
                add(rulecount, parse)
                bank.append([parse])

    rulecount['P'] = total_p

    for i in range(n_iterations):
        random_bin = numpy.random.random_integers(0, 1)
        LL_old = numpy.sum([rulecount[i]*numpy.log(rulecount[i]) for i in rulecount if rulecount[i] > 0], axis=0) - rulecount['P'] * numpy.log(rulecount['P']) - rulecount['S'] * numpy.log(rulecount['S'])

        rules = [rule for rule in rulecount.keys() if rule.count('P') > 1]

        temp_counts = dict(rulecount)
        temp_derivations = [list(bank[i]) for i in range(len(bank))]

        if random_bin:
            # Add a marker
            rules = [rule for rule in rulecount if rule.count('P') > 2 or (rule.count('P') == 2 and ( (rule[0] == 'S') or rule[-1] != 'P') )]
            chosen_rule_idx = numpy.random.randint(0, len(rules), size=1)
            chosen_rule = rules[chosen_rule_idx]

            usable_P = []
            shrunk_rule = chosen_rule[1:-1]
            shrunk_size = 1
            next_P = shrunk_rule.find('P')
            usable_P.append(next_P + shrunk_size)
            while True:
                shrunk_rule = shrunk_rule[next_P+1:]
                shrunk_size += next_P + 1
                next_P = shrunk_rule.find('P')
                if next_P == -1:
                    break
                usable_P.append(next_P + shrunk_size)
            chosen_P = usable_P[numpy.random.randint(0, len(usable_P))]

            for derivation_idx, derivation in enumerate(bank):
                for rule_idx, rule in enumerate(derivation):
                    if rule == chosen_rule:
                        temp_counts['P'] += 1
                        temp_counts[chosen_rule] -= 1

                        temp_counts[chosen_rule[:chosen_P+1]] = temp_counts[chosen_rule[:chosen_P+1]] + 1 if chosen_rule[:chosen_P+1] \
                        in temp_counts else 1
                        temp_counts[chosen_rule[chosen_P:]] = temp_counts[chosen_rule[chosen_P:]] + 1 if chosen_rule[chosen_P:] \
                        in temp_counts else 1

                        temp_derivations[derivation_idx][rule_idx] = chosen_rule[chosen_P:]
                        temp_derivations[derivation_idx].insert(rule_idx, chosen_rule[:chosen_P+1])

                        if temp_counts[chosen_rule] < 0:
                            print 'Hey Negative Count Encountered'
                        break
        else:
            # Remove a marker
            rules_end_with_P = [rule for rule in rulecount if rule[-1] == 'P']
            rules_begin_with_P = [rule for rule in rulecount if rule[0] == 'P']

            chosen_rules = []
            chosen_rule_idx = numpy.random.randint(0, len(rules_end_with_P), size=1)
            chosen_rule = rules_end_with_P[chosen_rule_idx]
            chosen_rules.append(chosen_rule)

            chosen_rule_idx = numpy.random.randint(0, len(rules_begin_with_P), size=1)
            chosen_rule = rules_begin_with_P[chosen_rule_idx]
            chosen_rules.append(chosen_rule)

            merged_rule = chosen_rules[0][:-1] + chosen_rules[1]

            for derivation_idx, derivation in enumerate(bank):
                for rule_idx, rule in enumerate(derivation):
                    if rule_idx < len(derivation)-1 and rule == chosen_rules[0] and derivation[rule_idx+1] == chosen_rules[1]:
                        temp_counts['P'] -= 1
                        temp_counts[chosen_rules[0]] -= 1
                        temp_counts[chosen_rules[1]] -= 1

                        temp_counts[merged_rule] = temp_counts[merged_rule] + 1 if merged_rule in temp_counts else 1

                        del temp_derivations[derivation_idx][rule_idx+1]
                        temp_derivations[derivation_idx][rule_idx] = merged_rule

                        break

        LL_new = numpy.sum([temp_counts[i]*numpy.log(temp_counts[i]) for i in temp_counts if temp_counts[i] > 0], axis=0) - temp_counts['P'] * numpy.log(temp_counts['P']) - temp_counts['S'] * numpy.log(temp_counts['S'])
        if LL_new > LL_old or (numpy.log(numpy.random.sample()) < (LL_new - LL_old)):
            rulecount = temp_counts
            bank = temp_derivations

    print 'Done'
    print rulecount

    sorted_rulecount = sorted(rulecount.items(), key=operator.itemgetter(1))
    print sorted_rulecount

    rulecount_S = {key: value for (key, value) in rulecount.items() if key[0] == 'S' and len(key) > 1}
    rulecount_P = {key: value for (key, value) in rulecount.items() if key[0] == 'P' and len(key) > 1}

    generated_numbers = dict()

    for i in range(n_samples):
        number = ''
        rule = drawRule(rulecount_S)
        number = rule[2::3]

        if rule[-1] == 'P':
            rule = drawRule(rulecount_P)
            number += rule[2::3]
            while rule[-1] == 'P':
                rule = drawRule(rulecount_P)
                number += rule[2::3]

        number = int(number)
        add(generated_numbers, number)

    keylist = range(limit)
    valuelist = []
    for i in keylist:
        if i in generated_numbers:
            valuelist.append(generated_numbers[i])
        else:
            valuelist.append(0)

    plt.bar(keylist, valuelist, color='g')
    plt.show()