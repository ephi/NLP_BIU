from collections import defaultdict
from docopt import docopt
import sys
import random


class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert (isinstance(lhs, str))
        assert (isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w, l, r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l, r, w)
        return grammar

    def is_terminal(self, symbol):
        return symbol not in self._rules

    def gen(self, symbol):
        if self.is_terminal(symbol):
            return symbol
        else:
            expansion = self.random_expansion(symbol)
            return " ".join(self.gen(s) for s in expansion)

    def random_sent(self, inc_trees, number_to_generate=1):
        if not include_trees:
            return [self.gen("ROOT") for x in range(number_to_generate)]
        return [self.gen_tree_structure("ROOT") for x in range(number_to_generate)]

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r, w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

    def gen_tree_structure(self, symbol):
        if self.is_terminal(symbol):
            return symbol
        return "({} {})".format(symbol, ' '.join(self.gen_tree_structure(symbol) for symbol in
                                                 self.random_expansion(symbol)))


if __name__ == '__main__':
    doc = """Usage: generate.py <grammar> [-n N_OF_S] [-t]

            -h --help    show this
            -n N_OF_S    number of sentences to print [default: 1]
            -t           include tree structures to print

            """
    arguments = docopt(doc, version='generate 1.0')
    grammar_file = arguments["<grammar>"]
    n_of_s = int(arguments["-n"])
    include_trees = arguments["-t"]
    pcfg = PCFG.from_file(grammar_file)
    sentences = pcfg.random_sent(include_trees, n_of_s)
    for i in range(len(sentences)-1):
        print(sentences[i]+"\n")
    print(sentences[-1])
