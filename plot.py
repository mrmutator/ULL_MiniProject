from __future__ import division
from matplotlib import pyplot as plt

# read data

infile = open("results/test_results.txt", "r")
data = []
for line in infile:
    data.append(tuple(line.strip().split()))

infile.close()

x = [tup[0] for tup in data]

# plot accepted likelihood

accepted_ll = [tup[3] for tup in data]
candidate_ll = [tup[2] for tup in data]

plt.plot(x, accepted_ll, label="accepted", linewidth=4)
plt.plot(x, candidate_ll, color="r", label="candidate")
plt.legend()
plt.title("Likelihood during sampling")
plt.xlabel("iterations")
plt.ylabel("log-likelihood")
plt.show()



et_types = [tup[4] for tup in data]
et_tokens = [tup[5] for tup in data]


#plt.plot(x, et_types, label="types")
plt.plot(x, et_tokens, color="r", label="tokens")
plt.legend()
plt.title("Number of el. trees in dataset")
plt.xlabel("iterations")
plt.ylabel("number of elementary trees")
plt.show()


plt.plot(x, et_types, label="types")
plt.legend()
plt.title("Number of grammar rules")
plt.xlabel("iterations")
plt.ylabel("number of grammar rules")
plt.show()


counts = {"A": 0, "F": 0, "R": 0}

labels = [tup[1] for tup in data[1:]]

for l in labels:
    counts[l] += 1

ls = []
fracs = []

for l, c in counts.items():
    ls.append(l)
    print l, c / (len(data) - 1)
    fracs.append(c / (len(data) - 1))
plt.pie(fracs, labels=ls, startangle=90, autopct='%1.1f%%')
plt.title("Sampling acceptance rate")
plt.show()