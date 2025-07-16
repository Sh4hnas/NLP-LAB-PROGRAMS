import nltk
from nltk import CFG, PCFG
from nltk.parse.chart import ChartParser
from nltk.parse.viterbi import ViterbiParser

cfg_grammar= CFG.fromstring("""
    S -> NP VP
    NP -> Det N 
    VP -> V NP 
    Det -> 'the' | 'a'
    N -> 'cat' | 'dog'
    V -> 'chased' | 'slept'
    ProperNoun -> 'Alice' | 'Bob'
""")

pcfg_grammar= PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | ProperNoun [0.4]
    VP -> V NP [0.7] | V [0.3]
    Det -> 'the' [0.6] | 'a' [0.4]
    N -> 'cat' [0.5] | 'dog' [0.5]
    V -> 'chased' [0.5] | 'slept' [0.5]
    ProperNoun -> 'Alice' [0.5] | 'Bob' [0.5]
""")

sentence= "the dog chased a cat".split()

print("\n Constituency Parsing using ChartParser (CFG):\n")
chart_parser= ChartParser(cfg_grammar)
for tree in chart_parser.parse(sentence):
    tree.pretty_print()
 
print("\n Probabilistic Parsing using ViterbiParser (PCFG):\n")
viterbi_parser= ViterbiParser(pcfg_grammar)
for tree in viterbi_parser.parse(sentence):
    tree.pretty_print()
    print(" Log Probability:", tree.prob())
