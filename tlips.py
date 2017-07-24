# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:01:55 2017

@author: Rafael C. Carrasco
"""
__author__ = 'Rafael C. Carrasco'
__copyright__ = 'Universidad de Alicante, 2017'
__license__ = "GPL"
__version__ = "2.0"

import re
from random import uniform, seed
from math import log
from collections import deque, Counter

# A tree has a label (str) and children (list). 
# Children is None for a leaf tree (such as t = a)
class Tree(object):

    def __init__(self, label, children = None):
        self.label = label
        self.children = children
    
    # Return True if tree is a leaf (no children)    
    def is_leaf(self):
        return self.children == None
    
    # Return tuple of subtrees (empty for a leaf)
    def subtrees(self):
        if self.children:
            return self.children
        else:
            return tuple()
        
    # Return a string representation of the node such as 'a' or 'c(a b(d))'
    def __str__(self):
        if self.children:
            content = ' '.join(str(child) for child in self.children)
            return self.label + '(' + content + ')'
        else:
            return self.label

    # Parse the string representation of a tree or subtree
    @staticmethod         
    def parse_tree(s):
        s = s.strip() # Neglect neighbouring whitespace
        m = re.match('\w*', s)
        label = m.group(0)
        
        if len(label) == len(s): # leaf node
            children = None
        else:                   # internal node: remove brackets
            children = Tree.parse_forest(s[len(label) + 1:-1])
        
        return Tree(label, children)
    
    # Parse the string representation of the content of a node (child subtrees)
    @staticmethod    
    def parse_forest(s):
        forest = list()
        s = s.strip()
        depth = 0
        start = 0
        for pos in range(len(s)):
            if s[pos] == '(':
                depth += 1
            elif s[pos] == ')':
                depth -= 1
            elif s[pos] == ' ':
                if depth == 0:
                    node = Tree.parse_tree(s[start:pos])
                    forest.append(node)
                    start = pos + 1
        if start <  len(s):
            node = Tree.parse_tree(s[start:])
            forest.append(node)
        
        return forest
                    

# Test tree implementation
s = 'aa(b(e(g xh(a) i) ex) cx(d))'
t =  Tree.parse_tree(s)
print('t=', t)
assert(str(t)==s)

# A transition rule is lhs -> rhs with an associated weight (probability)
# lhs (str or int), rhs (tuple), weight(int or float)
# the rhs tuple starts with a label (str) followed by a list of arguments
class Rule(object):
    
    def __init__(self, lhs, rhs, weight):
        self._lhs = lhs
        self._rhs = rhs
        self._weight = weight
        
    def label(self):
        return self._rhs[0]
    
    def args(self):
        return self._rhs[1:]
        
    def lhs(self):
        return self._lhs
    
    def rhs(self):
        return self._rhs
    
    def weight(self):
        return self._weight
    
    def arity(self):
        return len(self._rhs) - 1
        
    def add_to_weight(self, n):
        self._weight += n
        
    def __str__(self):
        args = ' '.join(map(str, self.args()))
        return str(self.lhs()) + ' <- ' + self.label() + '(' + args + ')'


# A bottom-up deterministic tree automaton with a single final state (axiom),
# a set of states and a list of transtion rules.     
class DTA(object):
    
    # Provide axiom and list of transition rules
    def __init__(self, axiom, rules = []):
        self.axiom = axiom
        self.states = set(rule.lhs() for rule in rules) # No useless states expected
        self.rules = rules
        
        # Auxiliary data structures for quick access
        self.transitions = {rule.rhs():rule for rule in rules}
        self.rights = {state:set() for state in self.states}
        self.lefts = {state:set() for state in self.states}
        self.total = Counter()
        print('Q=', self.states)
        for rule in rules:
            self.lefts[rule.lhs()].add(rule)
            self.total[rule.lhs()] += rule.weight()
            for state in rule.args():
                self.rights[state].add(rule)

        
    # Return a string representation of the DTA
    def __str__(self):
        rules = ', '.join(map(str, self.rules))
        return 'S = ' + str(self.axiom) + '\nR = [' + rules + ']'
       
    # Generate a random subtree with output q (a DTA state)
    def _gen(self, q):
        z = uniform(0, 1) * self.total[q]
        s = 0
        for rule in self.lefts[q]:
            s += rule.weight()
            if z <= s:
                label = rule.label()
                if rule.arity() > 0:
                   children = [self._gen(state) for state in rule.args()]
                   return Tree(label, children)
                else:    
                   return Tree(label)
        
        return None
        
    # Generate a random tree using DTA probabilites (rule weights)
    def gen(self):
        return self._gen(self.axiom)
        
    # Return the output delta(t) for a tree or subtree t
    def delta(self, tree):
        key = ((tree.label),) + tuple(self.delta(s) for s in tree.subtrees())
        if key in self.transitions:
            return self.transitions[key].lhs()
        else:
            return None
            
        
    # Count number of occurrences when the DTA operates on a subtreee          
    def _add_tree_counts(self, tree):
        key = ((tree.label),) + tuple(self.delta(s) for s in tree.subtrees())
        if key in self.transitions:
            rule = self.transitions[key]
            lhs = rule.lhs()
            self.total[lhs] += 1
            rule.add_to_weight(1)
            return lhs
        else:
            return None
        
    # Compute weights from sample (maximum likelihood estimation)
    def compute_weights(self, sample):
        for rule in self.transitions.values():
            rule.weight = 0
        for tree in sample:
            self._add_tree_counts(tree)
            

# A DTA which accepts a sample of trees. 
# State 0 (the axiom) is the only final state   
class Acceptor(object):
    
    def __init__(self, trees):
        self.axiom = 0
        self.states = {self.axiom}
        self.rules = list()
        
        # Create indices
        self.transitions = dict()
        self.lefts = {0:set()}
        self.rights = {0:set()}
        self.total = Counter()
        for tree in trees:
            # add transitions
            root = self.add_tree(tree)
            # add root-transition
            self.add_final(root)

        # Auxiliary sets for the inference process
        self.kern = list()
        self.frontier = deque()
        self.merged = dict()
        self.kern_rules = set()

    # Return number of states
    def __len__(self):
        return len(self.states)
    
    # add a new state
    def add_state(self, state):
        if state not in self.states:
            self.states.add(state)
            self.rights[state] = set()
            self.lefts[state] = set() 
            
    # Add a final (accepting) state
    def add_final(self, state):
        self.add_state(state)
        key = ('ROOT', state)
        if key in self.transitions:
            self.transitions[key].add_to_weight(1)
        else:
            rule = Rule(0, key, 1)
            self.add_rule(rule)
     
    # Add a new rule to the aceptor                   
    def add_rule(self, rule):
        self.rules.append(rule)
        self.transitions[rule.rhs()] = rule
        lhs = rule.lhs()
        self.add_state(lhs)
        self.lefts[lhs].add(rule)
        self.total[lhs] += rule.weight()
        for state in rule.args():
            self.add_state(state)
            self.rights[state].add(rule)
                
        
    def __str__(self):
        rules = '\n'.join(str(r) + ':' + str(r.weight()) for r in self.rules)
        return 'S = ' + str(self.axiom) \
                + '\nQ = ' + str(self.states) \
                + '\nR=\n' + rules 
        
    # Add all rules in a subtree
    def add_tree(self, tree):
        key = ((tree.label),) + tuple(self.add_tree(s) for s in tree.subtrees())
        if key in self.transitions:
            rule = self.transitions[key]
            rule.add_to_weight(1)
            self.total[rule.lhs()] += 1
            return rule.lhs()
        else: 
            # new state is required
            state = len(self.states)
            rule = Rule(state, key, 1)
            self.add_rule(rule)

            return state
            
    # Compare two Bernoulli outcomes: n1 out of t1 and n2 out of t2
    @staticmethod
    def differ(n1, t1, n2, t2, gamma):
        if t1 > 0 and t2 > 0:
            delta = abs(n1 / t1 - n2 / t2)
            err = gamma * (1 / t1 + 1 / t2) ** 0.5
            return delta > err
        else:
            return False
        
    # Check if kern_state is compatible with this state
    def compatible(self, kern_state, state, gamma):
        t1 = self.total[kern_state]
        t2 = self.total[state]
        for rule in self.rights[state]:
            n2 = rule.weight()
            rhs2 = rule.rhs()
            for pos in range(1, len(rhs2)):
                if rhs2[pos] == state:
                    rhs1 = list(rhs2)
                    rhs1[pos] = kern_state
                    key = tuple(rhs1)
                    if key in self.transitions:
                        kern_rule = self.transitions[key]
                        n1 = kern_rule.weight()
                    else:
                        n1 = 0
                    if self.differ(n1, t1, n2, t2, gamma):
                        #print(kern_state, state, n1, t1, n2, t2)
                        return False
                        if not self.compatible(kern_rule.lhs(), rule.lhs(), gamma):
                            return False
        
        for kern_rule in self.rights[kern_state]:
            n1 = kern_rule.weight()
            rhs1 = kern_rule.rhs()
            for pos in range(1, len(rhs1)):
                if rhs1[pos] == kern_state:
                    rhs2 = list(rhs1)
                    rhs2[pos] = state
                    key = tuple(rhs2)
                    if key in self.transitions:
                        rule = self.transitions[key]
                        n2 = rule.weight()  
                    else:
                        n2 = 0
                    if self.differ(n1, t1, n2, t2, gamma):
                        #print(kern_state, state, n1, t1, n2, t2)
                        return False
                        if not self.compatible(kern_rule.lhs(), rule.lhs(), gamma):
                          return False    
        
        return True
        
        
    def first_compatible_state_in_kern(self, state, gamma):
        #print(state, self.kern)
        for kern_state in self.kern:
            #print(kern_state)
            if kern_state > 0 and self.compatible(kern_state, state, gamma):
                #print('HELLO', kern_state)
                return kern_state
                
        return None
   
    # Add to kern and update frontier with lhs-states if rhs is fully in kern
    def add_to_kern(self, state):
        self.kern.add(state)
        for rule in self.rights[state]:
            #print('rule=', rule)
               if set(rule.args()) <= self.kern:
                   q = rule.lhs()
                   if q not in self.kern and q not in self.merged:
                       self.frontier.append(q)
                       self.kern_rules.add(rule)
       
        
    # The key algorithm
    # alpha(float) = significance level
    def infer(self, alpha):
        gamma = (0.5 * log(2 / alpha * len(self.rules)))
        self.kern = {0}
        self.frontier = deque(rule.lhs() for rule in self.rules if rule.arity() == 0)
        self.kern_rules = {rule for rule in self.rules if rule.arity() == 0}
        print('F=', self.frontier)
        while len(self.frontier) > 0: 
            #print('F =', self.frontier, 'K=', self.kern)
            state = self.frontier.popleft()
            kern_state = self.first_compatible_state_in_kern(state, gamma)
            if kern_state != None:
                self.merged[state] = kern_state
            else:
                #print('addded', state, 'to K = ', self.kern)
                self.add_to_kern(state)
                
        rules = self.projected_kern_rules()
        print('Merged', len(self.states), 'states into',
              len(self.kern) - 1, 'states and', len(rules), 'rules')  
        
        
        return str(list(self.kern)), rules
        
    def projected_kern_rules(self):
        rules = set(self.kern_rules)
        for rule in rules:
            if rule.lhs() in self.merged:
                rule._lhs = self.merged[rule.lhs()]
       
        return [str(rule) for rule in rules]
            
# Main code      
G = DTA(1, 
        [Rule(1, ('b',), 0.5),
         Rule(1, ('', 2, 2), 0.5),
         Rule(2, ('a',), 0.8),
         Rule(2, ('', 1, 1), 0.2)
        ])
            
#print(G)


t = Tree.parse_tree('((b (a ((a a) ((b (a a)) a)))) a)')
t = Tree.parse_tree('b')
print(t, ' -> ', G.delta(t))
assert(G.delta(t)==1)

seed(1)
sample = [G.gen() for n in range(1000)]

"""
with open('tlips.pkl', 'wb') as f:
    pickle.dump(sample, f)


with open('tlips.pkl', 'rb') as f:
    sample2 = pickle.load(f)
"""

a = Acceptor(sample)
#print(a)
res = a.infer(1/len(sample))
print('Kern=', res[0])
print('Rules=', res[1])


G = DTA(1, 
    [Rule(1, ('if-then-else-endif', 2, 1, 1), 0.2),
     Rule(1, ('if-then-endif', 2, 1), 0.2),
     Rule(1, ('print', 2), 0.6), 
     Rule(2, ('operator', 2, 3), 0.3),
     Rule(2, ('', 3), 0.7),   
     Rule(3, ('exp n', 3), 0.1),
     Rule(3, ('', 4), 0.9),
     Rule(4, ('lpar-rpar', 3), 0.2),
     Rule(4, ('n',), 0.8),
    ])
    
sample = [G.gen() for n in range(500)]
#print('t=', [str(t) for t in sample])
a = Acceptor(sample)
#print(a)
res = a.infer(1/len(a))
print('Kern=', res[0])
print('Rules=', res[1])
