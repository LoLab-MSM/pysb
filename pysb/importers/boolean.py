from __future__ import print_function as _
try:
    from boolean2 import boolmodel
except ImportError:
    raise ImportError('BooleanTranslator requires the package booleannet. \
    Run "pip install git+https://github.com/ialbert/booleannet" at the command line to install it')
from pysb.builder import Builder
from pysb.core import *
import copy
import pydot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sympy import Piecewise
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

class BooleanTranslationError(Exception):
    pass

class _Node(): 
    """
    Node class for tree construction
    """
    def __init__(self, function, function_nodes, id='', index=1):
        self.id = id
        self.tree = None
        self.index = index
        self.index_name = None
        self.value = None
        self.function = function
        self.function_nodes = function_nodes
        self.true_node = None
        self.false_node = None
        self.true_function = None
        self.false_function = None
        self.reduced_node_set = None
        self.mark = None

    def getTrueNode(self):
        return self.true_node
    
    def getFalseNode(self):
        return self.false_node
            
    def insertTrue(self, true_function, reduced_node_set, true_id, index):
        self.true_node = _Node(true_function, reduced_node_set, true_id, index)
        
    def insertFalse(self, false_function, reduced_node_set, false_id, index):
        self.false_node = _Node(false_function, reduced_node_set, false_id, index)

class BooleanTranslator(Builder):
    """
    Assemble a Model from a BooleanNet model file.

    See :py:func:`model_from_boolean` for further details.
    """
    _supported_formats = ['BooleanNet']
    _supported_modes = ['GSP','ROA','GA','SYN']
    # GSP: Gillespie
    # ROA: Random-order asynchronous
    # GA:  General asynchronous    
    # SYN: Synchronous
    
    def __init__(self, filename, format='BooleanNet', mode='GSP', force=False):
        super(BooleanTranslator, self).__init__()
        if format not in self._supported_formats:
            raise BooleanTranslationError('\'%s\' is not a supported file format. Try one of %s.' % 
                                          (str(format), str(self._supported_formats)))
        elif mode not in self._supported_modes:
            raise BooleanTranslationError('\'%s\' is not a supported update mode. Try one of %s.' % 
                                          (str(mode), str(self._supported_modes)))
        self._parse_input_file(filename, format=format)
        # create monomers, initial conditions, and observables
        #~~~~~
        if mode in ['ROA','SYN']:
            # patterns for triggering the RESET rule
            nfired_pat = []
        #~~~~~
        for name,state in self.initial_states.items():
            rank = self.function_ranks.get(name, 1)
            # create monomer
            mon_sites = ['state']
            mon_site_states = {'state' : ['False','True']}
            #~~~~~
            if mode in ['ROA','GA','SYN'] and rank > 1:
                mon_sites.append('delay')
                mon_site_states['delay'] = [str(i) for i in range(rank)]
            if mode in ['ROA', 'SYN']:
                mon_sites.append('reset')
                mon_site_states['reset'] = ['N','Y']
            if mode == 'SYN':
                mon_sites.append('copy')
                mon_site_states['copy'] = ['False','True','None']
                if rank > 1:
                    mon_site_states['copy'].append('Delay')
            #~~~~~
            mon = self.monomer(name, mon_sites, mon_site_states)
            #~~~~~
            # store pattern for triggering RESET rule
            if mode in ['ROA','SYN']:
                mon_pat_states = {'reset' : 'Y'}
                mon_pat = MonomerPattern(mon, mon_pat_states, compartment=None)
                nfired_pat.append(as_complex_pattern(mon_pat))
            #~~~~~
            # create initial condition and observable for both 'False' and 'True' states
            for s in mon.site_states['state']:
                # Observable
                mon_pat_states = {'state' : s}
                mon_pat = MonomerPattern(mon, mon_pat_states, compartment=None)
                self.observable('%s_%s_obs'%(name,s), as_complex_pattern(mon_pat))
                # Initial
                # redefine 'mon_pat_states' so that changes below don't affect observable
                mon_pat_states = {'state' : s} 
                #~~~~~
                if mode in ['ROA','GA','SYN'] and rank > 1:
                    mon_pat_states['delay'] = '0'
                if mode in ['ROA', 'SYN']:
                    mon_pat_states['reset'] = 'N'
                if mode == 'SYN':
                    mon_pat_states['copy'] = 'None'
                #~~~~~
                mon_pat = MonomerPattern(mon, mon_pat_states, compartment=None)
                init = self.parameter('%s_%s_init'%(name,s), 1 if s == state else 0)
                self.initial(as_complex_pattern(mon_pat), init)
        #~~~~~
        # RESET rule
        if mode in ['ROA', 'GA', 'SYN']:
            k_reset = self.parameter('k_reset', 1e10) 
            reset_mon = self.monomer('RESET', ['reset'], {'reset' : ['N', 'Y']})
            mon_pat = MonomerPattern(reset_mon, {'reset' : 'N'}, compartment=None)
            self.initial(as_complex_pattern(mon_pat), self.parameter('RESET_init', 1))
            reset_reac = MonomerPattern(reset_mon, {'reset' : 'Y'}, compartment=None)
            reset_prod = mon_pat
            is_reversible = False
            rate_forward = k_reset
            rate_reverse = None
            
            if mode in ['ROA', 'SYN']:
                # RESET(reset~N) <-> RESET(reset~Y)  1e10*if(N_FIRED>(N_NODES-0.5),1,0), 1e10*if(N_FIRED<0.5,1,0)
                is_reversible = True
                n_fired = self.observable('N_FIRED', ReactionPattern(nfired_pat))
                n_nodes = self.parameter('N_NODES', len(self.initial_states.keys()))
                rate_forward = self.expression('reset_Y_N', Piecewise((k_reset, n_fired < 0.5), (0, True))) #k_reset*(n_fired< 0.5))
                rate_reverse = self.expression('reset_N_Y', Piecewise((k_reset, n_fired > (n_nodes-0.5)), (0, True))) #  k_reset*(n_fired>(n_nodes-0.5)))
                        
            rule_expr = RuleExpression(as_reaction_pattern(reset_reac),
                                       as_reaction_pattern(reset_prod),
                                       is_reversible=is_reversible)
            self.rule('RESET_rule', rule_expr, rate_forward, rate_reverse)
        #~~~~~

        # minimize the ROBDD paths
        orderedNodes = self._findMinPathOrderHeap(self.functions, self.function_nodes)
        
        # create Rules
        BDDs = self._grove(self.functions, orderedNodes)
        for bdd in BDDs:
            self._createRules(bdd, mode=mode)
    
    def _parse_input_file(self, filename, format="BooleanNet"):
        if format not in BooleanTranslator._supported_formats:
            raise BooleanTranslationError(format + ' file type not supported')
        elif format == 'BooleanNet':
            self._parse_BooleanNet_file(filename)
        
    def _parse_BooleanNet_file(self, filename):
        """
        Parses a BooleanNet input file using the BooleanNet file parser.
        Creates and populates the instance dictionaries ``initial_states``, 
        ``functions``, ``function_nodes``, and ``function_ranks``.
        """
        text = open(filename, 'r').read()
        parser = boolmodel.BoolModel( mode='rank', text=text )
        parser.initialize()
        # Initial conditions
        self.initial_states = {} # initial states for each node
        for init in parser.init_tokens:
            state = init[-1].value
            for token in init:
                if token.type == 'ID':
                    self.initial_states[token.value] = state
        # Boolean functions
        self.functions = {} # tokenized lists of all Boolean functions
        self.function_nodes = {} # lists of nodes incident on each node (function variables)
        self.function_ranks = {} # ranks for each update function
        for rule in parser.update_tokens:
            node = rule[1].value
            self.functions[node] = []
            self.function_nodes[node] = []
            self.function_ranks[node] = rule[0].value
            for token in rule[2:]:
                if token.type != 'ASSIGN' and token.type != 'EQUAL':
                    self.functions[node].append(token.value)
                if token.type == 'ID' and token.value not in self.function_nodes[node]:
                    self.function_nodes[node].append(token.value)
    
    def _constructTree(self, current_node): 
        """
        Expands the tree via Shannon expansion and computes values of the leaves
        """
        if current_node.index == 1:
            current_node.tree = current_node.id
        nodes = current_node.function_nodes[:]
        id = current_node.id
        index = current_node.index
        function = current_node.function[:]
        function.insert(0, '(')
        function.append(')')
        function_update = []
        true_half = function[:]
        false_half = function[:]
        if nodes != []:
            current_node.index_name = current_node.function_nodes[0]
            for i,token in enumerate(function):
                if token == nodes[0]:
                    true_half[i] = 'True'
                    false_half[i] = 'False'
            true_half.extend(['and', nodes[0], 'or'])
            false_half.extend(['and', '( not', nodes[0], ')'])
            function_update = true_half + false_half
            true_function = function_update[:]
            false_function = function_update[:]
            for i,bool in enumerate(true_function):
                if bool == nodes[0]:
                    true_function[i] = 'True'
            for i,bool in enumerate(false_function):
                if bool == nodes[0]:
                    false_function[i] = 'False'                
            current_node.true_function = true_function
            current_node.false_function = false_function         
            current_node.reduced_node_set = nodes[1:]
            current_node.insertTrue(true_function, nodes[1:], id+'1', index+1)
            current_node.insertFalse(false_function, nodes[1:], id+'0', index+1)
            self._constructTree(current_node.true_node)
            self._constructTree(current_node.false_node)
        else:
            expression = " ".join(map(str, function))
            current_node.value = eval(expression)
            current_node.index_name = str(eval(expression))
    
    def _indexNodes(self, expansion, indexList=None):  
        """
        Creates index for use in constructing the ROBDD from the tree
        """
        if indexList == None:
            indexList = {}
        if expansion != None:
            if expansion.index in indexList and expansion not in indexList[expansion.index]:
                indexList[expansion.index].append(expansion)
                self._indexNodes(expansion.getTrueNode(), indexList)
                self._indexNodes(expansion.getFalseNode(), indexList)
            else:
                indexList[expansion.index] = []      
                indexList[expansion.index].append(expansion)
                self._indexNodes(expansion.getTrueNode(), indexList)
                self._indexNodes(expansion.getFalseNode(), indexList)
        return indexList
    
    def _constructBDD(self, treeRoot): 
        """
        Constructs the ROBDD
        """
        BDDroot = copy.deepcopy(treeRoot)
        indexList = self._indexNodes(BDDroot)
        L = len(indexList)
        
        # define leaf nodes
        True_leaf = _Node(None, None, '1', L)
        True_leaf.value = True
        True_leaf.index_name = 'True'  
        False_leaf = _Node(None, None, '0', L)
        False_leaf.value = False
        False_leaf.index_name = 'False'
    
        # set final node level to appropriate leaf node (True/False value)
        for i,j in enumerate(indexList[L-1]):
            if indexList[L-1][i].true_node.value == True:
                indexList[L-1][i].true_node = True_leaf
            else:
                indexList[L-1][i].true_node = False_leaf
            if indexList[L-1][i].false_node.value == True:
                indexList[L-1][i].false_node = True_leaf
            else:
                indexList[L-1][i].false_node = False_leaf  
        n=1
        while L-n != 1:
            
            # find redundant nodes
            for i,j in enumerate(indexList[L-n]):
                if indexList[L-n][i].true_node == indexList[L-n][i].false_node:
                    for k,l in enumerate(indexList[L-n-1]):
                        if indexList[L-n-1][k].true_node.id == indexList[L-n][i].id:
                            indexList[L-n-1][k].true_node = indexList[L-n][i].true_node
                            indexList[L-n][i].mark = 'redundant'
                        if indexList[L-n-1][k].false_node.id == indexList[L-n][i].id:
                            indexList[L-n-1][k].false_node = indexList[L-n][i].true_node
                            indexList[L-n][i].mark = 'redundant'
                            
            # remove redundant nodes
            removeList = []
            for i,j in enumerate(indexList[L-n]):
                if indexList[L-n][i].mark == 'redundant':
                    removeList.append(indexList[L-n][i])
            for each in removeList:
                indexList[L-n].remove(each)
                
            # find isomorphic nodes
            dup = []
            removeList = []
            for i,j in enumerate(indexList[L-n]):
                if indexList[L-n][i].mark == None:
                    indexList[L-n][i].mark = indexList[L-n][i].id
                    dup.append(indexList[L-n][i].mark)
                    for k,l in enumerate(indexList[L-n]):
                        if (indexList[L-n][k].mark == None) and (indexList[L-n][i].true_node == indexList[L-n][k].true_node) and (indexList[L-n][i].false_node == indexList[L-n][k].false_node):
                            indexList[L-n][k].mark = indexList[L-n][i].id
                            removeList.append(indexList[L-n][k])
            
            # combine isomorphic nodes     
            for each in dup:
                marked = False
                temp_node = None
                for i,j in enumerate(indexList[L-n-1]):
                    if indexList[L-n-1][i].true_node.mark == each and marked == False:
                        temp_node = indexList[L-n-1][i].true_node
                        marked = True
                    if indexList[L-n-1][i].false_node.mark == each and marked == False:
                        temp_node = indexList[L-n-1][i].false_node
                        marked = True
                                           
                    if indexList[L-n-1][i].true_node.mark == each and marked == True:
                        indexList[L-n-1][i].true_node = temp_node
                    if indexList[L-n-1][i].false_node.mark == each and marked == True:
                        indexList[L-n-1][i].false_node = temp_node
            for each in removeList:
                indexList[L-n].remove(each)
            n += 1
            
        return BDDroot
    
    def _computeTruthTable(self, function, nodes): 
        """Computes a truth table for a Boolean function and a set of nodes.
           Gives us the initial leaf ordering for :py:func:`_findMinPathOrderHeap`.
        """
        header = copy.deepcopy(nodes)
        header.append('result')
        table = []
        for i in range(2**len(nodes)):
            table.append([])
        for i,node in enumerate(nodes, 1):
            k = len(table)/(2**i)
            count = 1
            value = True
            for j,case in enumerate(table, 1):
                case.append(value)
                if count == k:
                    count = 0
                    value = not value
                count+=1
             
        for i,each in enumerate(table):
            function_copy = copy.deepcopy(function)
            for j,node in enumerate(nodes):
                for k,token in enumerate(function_copy):
                    if token == node:
                        function_copy[k] = each[j]        
            expression = " ".join(map(str, function_copy))
            value = eval(expression)
            table[i].append(value)       
        table.insert(0, header)
        return table
    
    def _leafSwap(self, nodes, leaves, high, low): 
        """
        Rearranges leaves in accordance to the new node order
        """
        lenN = len(nodes)
        lenL = len(leaves)
        groupSize = 2**(lenN-high+1)
        exchangeSize = 2**(lenN-low)
        ind = 0
        for i in range(int(lenL/groupSize)):
            for j in range(0, int(groupSize/2), exchangeSize):
                if ind == 0:
                    ind = 1
                elif ind == 1:
                    ind = 0
                    leaves[(i*groupSize + j):(i*groupSize + j)+exchangeSize], \
                    leaves[(i*groupSize + j)+int(groupSize/2)-exchangeSize:(i*groupSize + j)+int(groupSize/2)] = \
                    leaves[(i*groupSize + j)+int(groupSize/2)-exchangeSize:(i*groupSize + j)+int(groupSize/2)], \
                    leaves[(i*groupSize + j):(i*groupSize + j)+exchangeSize]
    
        return leaves   
    
    def _findMinPathOrderHeap(self, functions, nodes): 
        """
        Determines a variable order for the minimum number of BDD paths using Heap's algorithm; this is a brute force method 
        """
        newNodes = {}
        for key in functions:
            # initial path reduction count
            newNodes[key] = copy.deepcopy(nodes[key])
            N = len(nodes[key])
            leaves = []
            table = self._computeTruthTable(functions[key], nodes[key])
            for i in range(1, len(table)):
                leaves.append(table[i][-1])
            lenLeaves = len(leaves)
            marks = [1]*lenLeaves
            for each in range(N):
                set_size = 2**each
#                 num_sets = 2**(N-each) 
                counter = 1
                left = None
                right = None
                for j in range(0, lenLeaves, set_size):
                    if counter == 1:
                        left = leaves[j:j+set_size]
                        counter = 2
                    else:
                        right = leaves[j:j+set_size]
                        if left == right:
                            for k in range(j, j+set_size):
                                marks[k] = 0
                        counter = 1
            pathcount = 0
            for each in marks:
                pathcount += each
            index = [0]*N
            i = 1
            
            # run through all permutations of node order while counting path reduction
            while i < N:
                if index[i] < i:
                    swap = i % 2 * index[i]
                    leaves = self._leafSwap(nodes[key], leaves, swap+1, i+1)
                    marks = [1]*(lenLeaves)
                    for each in range(N):
                        set_size = 2**each
#                         num_sets = 2**(N-each)
                        counter = 1
                        left = None
                        right = None       
                        for j in range(0, lenLeaves, set_size):
                            if counter == 1:
                                left = leaves[j:j+set_size]
                                counter = 2
                            else:
                                right = leaves[j:j+set_size]
                                if left == right:
                                    for k in range(j, j+set_size):
                                        marks[k] = 0
                                counter = 1
                    nodes[key][swap], nodes[key][i] = nodes[key][i], nodes[key][swap]
                    paths = 0
                    for each in marks:
                        paths += each
                    if paths < pathcount:
                        pathcount = paths
                        newNodes[key] = copy.deepcopy(nodes[key])
                    index[i] += 1
                    i = 1
                else:
                    index[i] = 0
                    i+= 1
                    
        return newNodes
    
    def _pathExpansion(self, expansion, path=None, paths=None):    
        """
        Lists the paths for a given BDD or tree
        """
        if path is None:
            path = []
        if paths is None:
            paths = []
        if expansion is not None:
            if expansion.value is not None:
                path.append(expansion.index_name)
                paths.append(path)
            else:
                path.append(expansion.index_name)
                path_t = copy.deepcopy(path)
                path_f = copy.deepcopy(path)
                path_t.append('True')
                self._pathExpansion(expansion.getTrueNode(), path_t, paths)
                path_f.append('False')
                self._pathExpansion(expansion.getFalseNode(), path_f, paths)
                
        return paths
    
    def _createRules(self, root, mode='GSP'): 
        """
        Create Rules from the paths in a BDD (or tree)
        """
        function_node = root.tree
        rank = self.function_ranks[function_node]
        k_rate = 1./rank
        #~~~~~
        if mode in ['ROA','GA','SYN']:
            k_rate = 1.
            reset_mon = self.model.monomers['RESET']
            reset_pat_N = as_complex_pattern(MonomerPattern(reset_mon, {'reset' : 'N'}, compartment=None))
            reset_pat_Y = as_complex_pattern(MonomerPattern(reset_mon, {'reset' : 'Y'}, compartment=None))
        #~~~~~
        rate_expr = self.parameter('k_rate_%s' % function_node, k_rate)
        paths = self._pathExpansion(root)
        n = 0 # number of rules
        for p in paths:
            skip = False
            #~~~~~
            if mode in ['ROA','GA','SYN']:
                clipped = False
            #~~~~~
            if function_node in p: # node update depends on itself
                idx = p.index(function_node)
                if p[idx+1] == p[-1]: # this is a null event (not state change)
                    skip = True
                else:
                    p = p[:idx] + p[idx+2:] # remove node + state
                    #~~~~~
                    if mode in ['ROA','GA','SYN']:
                        clipped = True
                    #~~~~~
            if not skip:
                mon = self.model.monomers[function_node]
                reactant_patterns = []
                product_patterns = []
                reac_states = {'state' : 'True'} if p[-1] == 'False' else {'state' : 'False'}
                prod_states = {'state' : 'False'} if p[-1] == 'False' else {'state' : 'True'}
                #~~~~~
                if mode in ['ROA','GA','SYN']:
                    if not clipped:
                        reac_states = {'state' : WILD}
                    if rank > 1:
                        reac_states['delay'] = str(rank-1)
                        prod_states['delay'] = '0'    
                if mode in ['ROA', 'SYN']:        
                    reac_states['reset'] = 'N'
                    prod_states['reset'] = 'Y'
                if mode == 'SYN':
                    reac_states['copy'] = 'None'
                    prod_states['copy'] = prod_states['state']
                    prod_states['state'] = reac_states['state']
                #~~~~~
                reac_pat = as_complex_pattern(MonomerPattern(mon, reac_states, compartment=None))
                prod_pat = as_complex_pattern(MonomerPattern(mon, prod_states, compartment=None))
                reactant_patterns.append(reac_pat)
                product_patterns.append(prod_pat)
                j = 0
                while j < len(p[:-1]):
                    if p[j] != function_node:
                        mon = self.model.monomers[p[j]]
                        node_context = as_complex_pattern(MonomerPattern(mon, {'state' : p[j+1]}, 
                                                                         compartment=None))
                        reactant_patterns.append(node_context)
                        product_patterns.append(node_context)
                    j = j + 2
                #~~~~~
                if mode in ['ROA','GA','SYN']:
                    reactant_patterns.append(reset_pat_N)
                    if mode == 'GA':
                        product_patterns.append(reset_pat_Y)
                    else:
                        product_patterns.append(reset_pat_N)
                #~~~~~
                rule_expr = RuleExpression(ReactionPattern(reactant_patterns),
                                          ReactionPattern(product_patterns),
                                          is_reversible=False)
                self.rule('%s_rule%d' % (function_node,n), rule_expr, rate_expr)
                n = n + 1
        #~~~~~
        # delay, copy, and reset rules
        if mode in ['ROA','GA','SYN']:
            mon = self.model.monomers[function_node]
            # delay rules
            for d in range(rank-1):
                reac_pat_states = {'delay' : str(d)}
                prod_pat_states = {'delay' : str(d+1)}
                if mode in ['ROA','SYN']:
                    reac_pat_states['reset'] = 'N'
                    prod_pat_states['reset'] = 'Y'
                if mode == 'SYN':
                    reac_pat_states['copy'] = 'None'
                    prod_pat_states['copy'] = 'Delay'
                reac_pat = [as_complex_pattern(MonomerPattern(mon, reac_pat_states, compartment=None))]
                prod_pat = [as_complex_pattern(MonomerPattern(mon, prod_pat_states, compartment=None))]
                reac_pat.append(reset_pat_N)
                if mode == 'GA':
                    prod_pat.append(reset_pat_Y)
                else:
                    prod_pat.append(reset_pat_N)
                rule_expr = RuleExpression(ReactionPattern(reac_pat),
                                           ReactionPattern(prod_pat),
                                           is_reversible=False)
                self.rule('%s_delay_%d_%d' % (function_node,d,d+1), rule_expr, rate_expr)
            # copy rules
            if mode == 'SYN':
#                 mon = self.model.monomers[function_node]
                for s in ['False','True']:
                    reac_pat_states = {'state' : WILD, 'reset' : 'Y', 'copy' : s}
                    prod_pat_states = {'state' : s, 'reset' : 'Y', 'copy' : 'None'}
                    reac_pat = [as_complex_pattern(MonomerPattern(mon, reac_pat_states, compartment=None))]
                    prod_pat = [as_complex_pattern(MonomerPattern(mon, prod_pat_states, compartment=None))]
                    reac_pat.append(reset_pat_Y)
                    prod_pat.append(reset_pat_Y)
                    rule_expr = RuleExpression(ReactionPattern(reac_pat),
                                               ReactionPattern(prod_pat),
                                               is_reversible=False)
                    self.rule('%s_copy_%s' % (function_node,s), rule_expr, self.model.parameters['k_reset'])
                # need a copy rule for delays to ensure that in each round three rules fire (state change, copy, reset) 
                # for each species node (other than RESET)
                if rank > 1:
                    reac_pat = [as_complex_pattern(MonomerPattern(mon, {'reset' : 'Y', 'copy' : 'Delay'}, compartment=None))]
                    prod_pat = [as_complex_pattern(MonomerPattern(mon, {'reset' : 'Y', 'copy' : 'None'},  compartment=None))]
                    reac_pat.append(reset_pat_Y)
                    prod_pat.append(reset_pat_Y)
                    rule_expr = RuleExpression(ReactionPattern(reac_pat),
                                               ReactionPattern(prod_pat),
                                               is_reversible=False)
                    self.rule('%s_copy_Delay' % function_node, rule_expr, self.model.parameters['k_reset'])
            # reset rule(s)
            if mode in ['ROA','SYN']:
                reac_pat = [reset_pat_Y]
                prod_pat = [reset_pat_Y]
                reac_pat_states = {'reset' : 'Y'}
                prod_pat_states = {'reset' : 'N'}
                if mode == 'SYN':
                    reac_pat_states['copy'] = 'None'
                    prod_pat_states['copy'] = 'None'
                reac_pat.append(as_complex_pattern(MonomerPattern(mon, reac_pat_states, compartment=None)))
                prod_pat.append(as_complex_pattern(MonomerPattern(mon, prod_pat_states, compartment=None)))
                rule_expr = RuleExpression(ReactionPattern(reac_pat), ReactionPattern(prod_pat), is_reversible=False)
                self.rule('%s_reset' % function_node, rule_expr, self.model.parameters['k_reset'])
            #~~~~~
    
    def _grove(self, functions, function_nodes): 
        """ 
        Groups all BDDs (or trees)
        """
        tree_list = []
        bdd_list = []
        for keys,values in functions.items():
            tree_root = _Node(functions[keys], function_nodes[keys], keys, 1)
            nodeNum = len(function_nodes[keys])
            self._constructTree(tree_root)
            tree_list.append(tree_root)
            bdd_root = self._constructBDD(tree_root)
            bdd_list.append(bdd_root)
            
        return bdd_list
    
    def _displayIndex(self, expansion, indexList=None):  
        """
        Creates index for use in printTree
        """
        if indexList is None:
            indexList = {}
        if expansion != None:
            if expansion.index in indexList:        
                present = False
                for j in indexList[expansion.index]:
                    if j == expansion:
                        present = True
                if present == False:
                    indexList[expansion.index].append(expansion)
                    self._displayIndex(expansion.getTrueNode(), indexList)
                    self._displayIndex(expansion.getFalseNode(), indexList)
            else:
                indexList[expansion.index] = []                
                present = False
                for j in indexList[expansion.index]:
                    if j == expansion:
                        present = True
                if present == False:
                    indexList[expansion.index].append(expansion)
                    self._displayIndex(expansion.getTrueNode(), indexList)
                    self._displayIndex(expansion.getFalseNode(), indexList)
                    
        return indexList
    
    def printTree(self, expansion, fname='BDD', show=False): 
        """
        Utility for displaying the ROBDD
        
        Usage
        -----
        printTree(_constructBDD(Node(function, node_list)))
        """
        indexList = self._displayIndex(expansion)
        graph = pydot.Dot(graph_type='digraph')
        for i,j in enumerate(indexList, 1):
            for k,l in enumerate(indexList[i]):
                node = pydot.Node(indexList[i][k].index_name+indexList[i][k].id)
                node.set('label', indexList[i][k].index_name)
                node.set('rank', indexList[i][k].index)
                graph.add_node(node)
            for k,l in enumerate(indexList[i]):
                if i != len(indexList):
                    graph.add_edge(pydot.Edge(pydot.Node(indexList[i][k].index_name+indexList[i][k].id), 
                                              pydot.Node(indexList[i][k].true_node.index_name+indexList[i][k].true_node.id), 
                                              label='1'))
                    graph.add_edge(pydot.Edge(pydot.Node(indexList[i][k].index_name+indexList[i][k].id), 
                                              pydot.Node(indexList[i][k].false_node.index_name+indexList[i][k].false_node.id), 
                                              label='0'))
    
        func_string = expansion.id + '* = ' + ' '.join(expansion.function)
        node_order = 'order: ' + ', '.join(expansion.function_nodes)
        n_nodes = len(graph.get_nodes())-2
        n_paths = len(self._pathExpansion(expansion))
        nodes_and_paths = '%d nodes, %d paths' % (n_nodes,n_paths)
        plt.title( '%s \n %s \n %s' % (func_string, node_order, nodes_and_paths ) )
        png = graph.create_png(prog='dot')
        sio = StringIO()
        sio.write(png)
        sio.seek(0)
        img = mpimg.imread(sio)
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%s.pdf' % fname, dpi=600, format='pdf')
        if show:
            plt.show()
        
def model_from_boolean(filename, format='BooleanNet', mode='GSP', force=False):
    """
    Convert a Boolean model into a PySB Model.

    Notes
    -----

    Currently only BooleanNet model files are supported. Future versions may
    extend support to other formats, such as sbml-qual.

    Parameters
    ----------
    filename : string
        File containing the Boolean model
    force : bool, optional
        The default, False, will raise an Exception if there are any errors
        importing the model to PySB, e.g. due to unsupported features.
        Setting to True will attempt to ignore any import errors, which may
        lead to a model that only poorly represents the original. Use at own
        risk!
    """
    bt = BooleanTranslator(filename, format=format, mode=mode, force=force)
    return bt.model
