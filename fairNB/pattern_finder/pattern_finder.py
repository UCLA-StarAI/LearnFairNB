from pfwrapper import PyPatternFinder
from collections import namedtuple

Pattern = namedtuple('Pattern', ['base', 'sens', 'pBase', 'pAll',
                                 'pDX', 'pD_X', 'pX', 
                                 'pDY', 'pD_Y', 'pY', 
                                 'pDXY', 'pD_XY', 'pXY'])

DivergentPattern = namedtuple('DivergentPattern', 
                             ['base', 'sens',
                              'pDY', 'pD_Y',
                              'pDXY', 'pD_XY',
                              'score'])

class PatternFinder:
    def __init__(self, root_params, leaf_params,
                 target_value, sensitive_var_ids):
        self.pattern_finder = PyPatternFinder(root_params, leaf_params,
                                              target_value, sensitive_var_ids)
        self.num_visited = -1

    def convert_pattern(self, p):
        return DivergentPattern(base=p['base'], sens=p['sens'], 
                                pDY=p['pDY'], pD_Y=p['pNotDY'],
                                pDXY=p['pDXY'], pD_XY=p['pNotDXY'],
                                score=p['kld'])

    def find_any_divergent(self, threshold, num_patterns):
        patterns, self.num_visited = self.pattern_finder.get_divergent_patterns(threshold, num_patterns, True)
        patterns = [self.convert_pattern(p) for p in patterns]
        return patterns

    def find_any_discriminating(self, threshold, num_patterns):
        patterns, self.num_visited = self.pattern_finder.get_discriminating_patterns(threshold, num_patterns, True)
        patterns = [self.convert_pattern(p) for p in patterns]
        return patterns        

    def get_divergent_patterns(self, threshold, num_patterns):
        patterns, self.num_visited = self.pattern_finder.get_divergent_patterns(threshold, num_patterns)
        patterns = [self.convert_pattern(p) for p in patterns]
        return patterns

    def get_discriminating_patterns(self, threshold, num_patterns):
        patterns, self.num_visited = self.pattern_finder.get_discriminating_patterns(threshold, num_patterns)
        patterns = [self.convert_pattern(p) for p in patterns]
        return patterns        


if __name__ == '__main__':
    root_params = [0.1, 0.9]
    leaf_params = [[0.3, 0.7, 0.1, 0.9],
                   [0.4, 0.6, 0.5, 0.5],
                   [0.2, 0.8, 0.3, 0.7],
                   [0.15, 0.85, 0.25, 0.75]]
    target_value = 1
    sensitive_var_ids = [0, 2]
    threshold = 0.1
    num_patterns = 3

    pf = PatternFinder(root_params, leaf_params,
                       target_value, sensitive_var_ids)

    dis_patterns = pf.get_discriminating_patterns(threshold, num_patterns)
    print(dis_patterns)
    print('visited nodes: %d\n'%pf.num_visited)

    div_patterns = pf.get_divergent_patterns(threshold, num_patterns)
    print(div_patterns)
    print('visited nodes: %d\n'%pf.num_visited)
