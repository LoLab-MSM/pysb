import collections
from pysb.core import ComplexPattern, MonomerPattern, Monomer, ReactionPattern
import copy


class SimplePatternMatcher(object):
    @classmethod
    def match_monomer_pattern(cls, pattern, candidate):
        if pattern.compartment:
            raise NotImplementedError

        if pattern.monomer is not candidate.monomer:
            return False

        for site, condition in pattern.site_conditions.items():
            if candidate.site_conditions[site] != condition:
                return False

        return True

    @classmethod
    def match_complex_pattern(cls, pattern, candidate):
        if pattern.compartment:
            raise NotImplementedError

        for pat_mp in pattern.monomer_patterns:
            if not any([cls.match_monomer_pattern(pat_mp, test_mp)
                        for test_mp in candidate.monomer_patterns]):
                return False

        return True

    @classmethod
    def match_reaction_pattern(cls, pattern, candidate):
        # The ComplexPatterns need to match one to one
        cplx_pats_accounted_for = []
        candidate_cplx_pats = copy.copy(candidate.complex_patterns)
        for pat_idx, pat_cp in enumerate(pattern.complex_patterns):
            if pat_idx not in cplx_pats_accounted_for and \
                    cls._complex_pattern_in_reaction_pattern(
                        pat_cp, candidate_cplx_pats):
                cplx_pats_accounted_for.append(pat_idx)

        return len(cplx_pats_accounted_for) == len(pattern.complex_patterns)

    @classmethod
    def _complex_pattern_in_reaction_pattern(cls, pattern, cand_cplx_pats):
        for i, cp in enumerate(cand_cplx_pats):
            if cls.match_complex_pattern(pattern, cp):
                cand_cplx_pats.pop(i)
                return True
        return False


def _monomers_from_pattern(pattern):
    if isinstance(pattern, ReactionPattern):
        return set.union(*[_monomers_from_pattern(cp)
                           for cp in pattern.complex_patterns])
    if isinstance(pattern, ComplexPattern):
        return set([mp.monomer for mp in pattern.monomer_patterns])
    elif isinstance(pattern, MonomerPattern):
        return {pattern.monomer}
    elif isinstance(pattern, Monomer):
        return {pattern}
    else:
        raise Exception('Unsupported pattern type: %s' % type(pattern))


class SpeciesPatternMatcher(object):
    """
    Match a pattern against a model's species list

    No support for models with compartments yet

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.bng import generate_equations
    >>> from pysb.pattern import SpeciesPatternMatcher
    >>> generate_equations(model)
    >>> spm = SpeciesPatternMatcher(model)

    Assign two monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> Bax4 = model.monomers['Bax4']
    >>> Bcl2 = model.monomers['Bcl2']

    Search using a Monomer

    >>> spm.match_species(Bax4)
    [Bax4(b=None), Bax4(b=1) % Bcl2(b=1), Bax4(b=1) % Mito(b=1)]
    >>> spm.match_species(Bcl2) # doctest:+NORMALIZE_WHITESPACE
    [Bax2(b=1) % Bcl2(b=1),
    Bax4(b=1) % Bcl2(b=1),
    Bcl2(b=None),
    Bcl2(b=1) % MBax(b=1)]

    Search using a MonomerPattern

    >>> spm.match_species(Bax4(b=1))
    [Bax4(b=1) % Bcl2(b=1), Bax4(b=1) % Mito(b=1)]
    >>> spm.match_species(Bcl2(b=1))
    [Bax2(b=1) % Bcl2(b=1), Bax4(b=1) % Bcl2(b=1), Bcl2(b=1) % MBax(b=1)]

    Search using a ComplexPattern

    >>> spm.match_species(Bax4(b=1) % Bcl2(b=1))
    [Bax4(b=1) % Bcl2(b=1)]
    >>> spm.match_species(Bax4() % Bcl2())
    [Bax4(b=1) % Bcl2(b=1)]
    """
    def __init__(self, model):
        self.model = model
        if not model.species:
            raise Exception('Model needs species list - run '
                            'generate_equations() first')

        self._species_cache = collections.defaultdict(set)
        for idx, sp in enumerate(model.species):
            if sp.compartment:
                raise NotImplementedError
            for mp in sp.monomer_patterns:
                if mp.compartment:
                    raise NotImplementedError
                self._species_cache[mp.monomer].add(idx)

    def match_species(self, pattern):
        if not isinstance(pattern, (Monomer, MonomerPattern, ComplexPattern)):
            raise ValueError('A Monomer, MonomerPattern or ComplexPattern is '
                             'required to match species')

        monomers = _monomers_from_pattern(pattern)
        shortlist = self._species_containing_monomers(monomers)

        # If pattern is a Monomer, we're done
        if isinstance(pattern, Monomer):
            return shortlist

        # If pattern is a MonomerPattern, check any specified sites
        if isinstance(pattern, MonomerPattern):
            new_shortlist = []
            for sp in shortlist:
                for mp in sp.monomer_patterns:
                    if SimplePatternMatcher.match_monomer_pattern(pattern, mp):
                        new_shortlist.append(sp)
            return new_shortlist

        # ComplexPattern
        else:
            return [sp for sp in shortlist if
                    SimplePatternMatcher.match_complex_pattern(pattern, sp)]

    def _species_containing_monomers(self, monomer_list):
        """
        Identifies species containing a list of monomers

        Parameters
        ----------
        monomer_list: list of Monomers
            A list of monomers with which to search the model's species

        Returns
        -------
        Model species containing all of the monomers in the list

        """
        sp_indexes = set.intersection(*[self._species_cache[mon] for mon in
                                        monomer_list])
        return [self.model.species[sp] for sp in sp_indexes]


class RulePatternMatcher(object):
    """
    Match a pattern against a model's species list

    No support for models with compartments yet. Methods are provided to
    match against rule reactants, products or both. Searches can be
    Monomers, MonomerPatterns, ComplexPatterns or ReactionPatterns.

    ### TODO: Match Parameters and Expressions
    ### TODO: Test on models with multiple binding sites/states

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.pattern import RulePatternMatcher
    >>> rpm = RulePatternMatcher(model)

    Assign some monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> AMito, mCytoC, mSmac, cSmac = [model.monomers[m] for m in \
                                       ('AMito', 'mCytoC', 'mSmac', 'cSmac')]

    Search using a Monomer

    >>> rpm.match_reactants(AMito) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) <>
        AMito(b=1) % mCytoC(b=1), kf20, kr20),
    Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20),
    Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) <>
        AMito(b=1) % mSmac(b=1), kf21, kr21),
    Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_products(mSmac) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) <>
        AMito(b=1) % mSmac(b=1), kf21, kr21)]

    Search using a MonomerPattern

    >>> rpm.match_reactants(AMito(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20),
    Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_rules(cSmac(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rule('inhibit_cSmac_by_XIAP', cSmac(b=None) + XIAP(b=None) <>
        cSmac(b=1) % XIAP(b=1), kf28, kr28)]

    Search using a ComplexPattern

    >>> rpm.match_reactants(AMito() % mSmac()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
        AMito(b=None) + ASmac(b=None), kc21)]

    >>> rpm.match_rules(AMito(b=1) % mCytoC()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) <>
        AMito(b=1) % mCytoC(b=1), kf20, kr20),
    Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20)]

    Search using a ReactionPattern

    >>> rpm.match_reactants(mCytoC() + mSmac())
    []

    >>> rpm.match_reactants(AMito() + mCytoC()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) <>
        AMito(b=1) % mCytoC(b=1), kf20, kr20)]

    >>> rpm.match_rules(AMito(b=1) % mCytoC()) # doctest:+NORMALIZE_WHITESPACE
    [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) <>
        AMito(b=1) % mCytoC(b=1), kf20, kr20),
    Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
        AMito(b=None) + ACytoC(b=None), kc20)]

    """

    def __init__(self, model):
        self.model = model

        self._reactant_cache = collections.defaultdict(set)
        self._product_cache = collections.defaultdict(set)

        for rule in model.rules:
            for cache, rp in ((self._reactant_cache, rule.reactant_pattern),
                              (self._product_cache, rule.product_pattern)):
                for cp in rp.complex_patterns:
                    if cp.compartment:
                        raise NotImplementedError
                    for mp in cp.monomer_patterns:
                        if mp.compartment:
                            raise NotImplementedError
                        cache[mp.monomer].add(rule.name)

    def match_reactants(self, pattern):
        return self._match_reaction_patterns(pattern, 'reactant')

    def match_products(self, pattern):
        return self._match_reaction_patterns(pattern, 'product')

    def match_rules(self, pattern):
        return [r for r in self.model.rules if
                r in self.match_reactants(pattern) or
                r in self.match_products(pattern)]

    def _match_reaction_patterns(self, pattern, reaction_side):
        if not isinstance(pattern, (Monomer, MonomerPattern, ComplexPattern,
                                    ReactionPattern)):
            raise ValueError('A Monomer, MonomerPattern, ComplexPattern or '
                             'ReactionPattern required to match rules')

        monomers = _monomers_from_pattern(pattern)

        if reaction_side == 'reactant':
            cache = self._reactant_cache

            def pat_fn(r):
                return r.reactant_pattern
        elif reaction_side == 'product':
            cache = self._product_cache

            def pat_fn(r):
                return r.product_pattern
        else:
            raise Exception('reaction_side must be "reactant" or "product"')

        shortlist = self._cache_containing_monomers(cache, monomers)

        # If pattern is a Monomer, we're done
        if isinstance(pattern, Monomer):
            return shortlist

        # If pattern is a MonomerPattern, check any specified sites
        if isinstance(pattern, MonomerPattern):
            new_shortlist = []
            for rule in shortlist:
                reaction_pattern = pat_fn(rule)
                if self._match_monomer_pattern_to_reaction_pattern(
                        pattern, reaction_pattern):
                    new_shortlist.append(rule)

            return new_shortlist

        elif isinstance(pattern, ComplexPattern):
            new_shortlist = []
            for rule in shortlist:
                reaction_pattern = pat_fn(rule)
                if self._match_complex_pattern_to_reaction_pattern(
                        pattern, reaction_pattern):
                    new_shortlist.append(rule)

            return new_shortlist

        else:
            return [rule for rule in shortlist if
                    SimplePatternMatcher.match_reaction_pattern(
                        pattern, pat_fn(rule))]

    @classmethod
    def _match_monomer_pattern_to_reaction_pattern(cls, pattern, test_pattern):
        for cp in test_pattern.complex_patterns:
            for mp in cp.monomer_patterns:
                if SimplePatternMatcher.match_monomer_pattern(pattern, mp):
                    return True
        return False

    @classmethod
    def _match_complex_pattern_to_reaction_pattern(cls, pattern, test_pattern):
        for cp in test_pattern.complex_patterns:
            if SimplePatternMatcher.match_complex_pattern(pattern, cp):
                return True
        return False

    def _cache_containing_monomers(self, cache, monomer_list):
        """
        Identifies rules containing a list of monomers

        Parameters
        ----------
        monomer_list: list of Monomers
            A list of monomers with which to search the model's rules

        Returns
        -------
        Model rules containing all of the monomers in the list

        """
        rule_names = set.intersection(*[cache[mon] for mon in
                                        monomer_list])
        return [r for r in self.model.rules if r.name in rule_names]


class ReactionPatternMatcher(object):
    """
    Match a pattern against a model's reactions list

    No support for models with compartments yet. Methods are provided to
    match against reaction reactants, products or both. Searches can be
    Monomers, MonomerPatterns, ComplexPatterns or ReactionPatterns.

    Examples
    --------

    Create a PatternMatcher for the EARM 1.0 model

    >>> from pysb.examples.earm_1_0 import model
    >>> from pysb.bng import generate_equations
    >>> from pysb.pattern import ReactionPatternMatcher
    >>> generate_equations(model)
    >>> rpm = ReactionPatternMatcher(model)

    Assign some monomers to variables (only needed when importing a model
    instead of defining one interactively)

    >>> AMito, mCytoC, mSmac, cSmac = [model.monomers[m] for m in \
                                       ('AMito', 'mCytoC', 'mSmac', 'cSmac')]

    Search using a Monomer

    >>> rpm.match_products(mSmac) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (<>):
        Reactants: {'__s15': mSmac(b=None), '__s45': AMito(b=None)}
        Products: {'__s47': AMito(b=1) % mSmac(b=1)}
        Rate: __s15*__s45*kf21 - __s47*kr21
        Rules: [Rule('bind_mSmac_AMito', AMito(b=None) + mSmac(b=None) <>
                AMito(b=1) % mSmac(b=1), kf21, kr21)]]

    Search using a MonomerPattern

    >>> rpm.match_reactants(AMito(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (>>):
        Reactants: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Products: {'__s45': AMito(b=None), '__s48': ACytoC(b=None)}
        Rate: __s46*kc20
        Rules: [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
                AMito(b=None) + ACytoC(b=None), kc20)],
     Rxn (>>):
        Reactants: {'__s47': AMito(b=1) % mSmac(b=1)}
        Products: {'__s45': AMito(b=None), '__s49': ASmac(b=None)}
        Rate: __s47*kc21
        Rules: [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
                AMito(b=None) + ASmac(b=None), kc21)]]

    >>> rpm.match_reactions(cSmac(b=1)) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (<>):
        Reactants: {'__s7': XIAP(b=None), '__s51': cSmac(b=None)}
        Products: {'__s53': XIAP(b=1) % cSmac(b=1)}
        Rate: __s51*__s7*kf28 - __s53*kr28
        Rules: [Rule('inhibit_cSmac_by_XIAP', cSmac(b=None) + XIAP(b=None) <>
                cSmac(b=1) % XIAP(b=1), kf28, kr28)]]

    Search using a ComplexPattern

    >>> rpm.match_reactants(AMito() % mSmac()) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (>>):
        Reactants: {'__s47': AMito(b=1) % mSmac(b=1)}
        Products: {'__s45': AMito(b=None), '__s49': ASmac(b=None)}
        Rate: __s47*kc21
        Rules: [Rule('produce_ASmac_via_AMito', AMito(b=1) % mSmac(b=1) >>
                AMito(b=None) + ASmac(b=None), kc21)]]

    >>> rpm.match_reactions(AMito(b=1) % mCytoC()) # doctest:+NORMALIZE_WHITESPACE
    [Rxn (<>):
        Reactants: {'__s14': mCytoC(b=None), '__s45': AMito(b=None)}
        Products: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Rate: __s14*__s45*kf20 - __s46*kr20
        Rules: [Rule('bind_mCytoC_AMito', AMito(b=None) + mCytoC(b=None) <>
                AMito(b=1) % mCytoC(b=1), kf20, kr20)],
     Rxn (>>):
        Reactants: {'__s46': AMito(b=1) % mCytoC(b=1)}
        Products: {'__s45': AMito(b=None), '__s48': ACytoC(b=None)}
        Rate: __s46*kc20
        Rules: [Rule('produce_ACytoC_via_AMito', AMito(b=1) % mCytoC(b=1) >>
                AMito(b=None) + ACytoC(b=None), kc20)]]
    """
    def __init__(self, model, species_pattern_matcher=None):
        self.model = model

        # In this cache, our caches map species to reactions
        self._reactant_cache = collections.defaultdict(set)
        self._product_cache = collections.defaultdict(set)

        if not species_pattern_matcher:
            self.spm = SpeciesPatternMatcher(model)

        for r_id, rxn in enumerate(model.reactions_bidirectional):
            for cache, species_ids in (
                    (self._reactant_cache, rxn['reactants']),
                    (self._product_cache, rxn['products'])):
                for sp_id in species_ids:
                    sp = model.species[sp_id]
                    if sp.compartment:
                        raise NotImplementedError
                    cache[sp].add(r_id)

    def match_reactants(self, pattern):
        return self._match_reactions_against_cache(pattern, 'reactant')

    def match_products(self, pattern):
        return self._match_reactions_against_cache(pattern, 'product')

    def match_reactions(self, pattern):
        return self._match_reactions_against_cache(pattern, 'both')

    def _match_reactions_against_cache(self, pattern, reaction_side):
        species = self.spm.match_species(pattern)

        rxn_ids = set()
        if reaction_side in ['reactant', 'both']:
            rxn_ids.update(*[self._reactant_cache[sp] for sp in species])

        if reaction_side in ['product', 'both']:
            rxn_ids.update(*[self._product_cache[sp] for sp in species])
        rxn_ids = list(rxn_ids)
        rxn_ids.sort()

        return [Reaction(self.model.reactions_bidirectional[rxn_id],
                         self.model) for rxn_id in rxn_ids]


class Reaction(object):
    __slots__ = ['_rxn_dict', 'reactants', 'products', 'rate', 'reversible',
                 'rules']
    """
    Store reactions in object form for pretty-printing
    """
    def __init__(self, rxn_dict, model):
        self._rxn_dict = rxn_dict
        self.reactants = collections.OrderedDict()
        self.products = collections.OrderedDict()
        for s_id, s in enumerate(model.species):
            if s_id in rxn_dict['reactants']:
                self.reactants['__s%d' % s_id] = s
            if s_id in rxn_dict['products']:
                self.products['__s%d' % s_id] = s
        self.rate = rxn_dict['rate']
        self.reversible = rxn_dict['reversible']
        self.rules = [model.rules[r] for r in rxn_dict['rule']]

    def __repr__(self):
        return 'Rxn (%s): \n    Reactants: %s\n    Products: %s\n    ' \
               'Rate: %s\n    Rules: %s' % (
                    '<>' if self.reversible else '>>',
                    self._repr_ordered_dict_like_dict(self.reactants),
                    self._repr_ordered_dict_like_dict(self.products),
                    self.rate,
                    self.rules
               )

    def __cmp__(self, other):
        try:
            return self._rxn_dict == other._rxn_dict
        except AttributeError:
            return False

    @staticmethod
    def _repr_ordered_dict_like_dict(ordered_dict):
        return '{%s}' % ', '.join(["'%s': %s" % (k, v) for k, v in
                                   ordered_dict.items()])
