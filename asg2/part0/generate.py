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

    def random_sent(self, number_to_generate=1):
        sentences = [self.gen("ROOT") for x in range(number_to_generate)]
        return sentences

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r, w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r


if __name__ == '__main__':
    doc = """Usage: generate.py <grammar> [-n N_OF_S]

            -h --help    show this
            -n N_OF_S    number of sentences to print [default: 1]

            """
    arguments = docopt(doc, version='generate 1.0')
    grammar_file = arguments["<grammar>"]
    n_of_s = int(arguments["-n"])
    pcfg = PCFG.from_file(grammar_file)
    sentences = pcfg.random_sent(n_of_s)
    for i in range(len(sentences)-1):
        print(sentences[i]+"\n")
    print(sentences[-1])
