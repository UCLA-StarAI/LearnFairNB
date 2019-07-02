from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "pf_src/naive_bayes.hpp":
  cdef cppclass NaiveBayes:
    NaiveBayes()
    NaiveBayes(const pair[double, double]&, const vector[vector[double]]&) except +

cdef extern from "pf_src/search.hpp":
  ctypedef struct DivergentPattern:
    vector[pair[size_t, size_t]] base
    vector[pair[size_t, size_t]] sens
    double pDY
    double pNotDY
    double pDXY
    double pNotDXY
    double kld

  cdef cppclass Search:
    Search()
    Search(const NaiveBayes&, size_t, double, const vector[size_t]&) except +
    vector[DivergentPattern] getDivergentPatterns(size_t, bint)
    vector[DivergentPattern] getDiscriminatingPatterns(size_t, bint)
    size_t getNumNodes()

cdef class PyPatternFinder:
  cdef size_t target_value
  cdef vector[size_t] sensitive_var_ids
  cdef NaiveBayes c_nb

  def __init__(self, const pair[double, double]& root_params,
                     const vector[vector[double]]& leaf_params,
                     size_t target_value,
                     const vector[size_t]& sensitive_var_ids):
    self.target_value = target_value
    self.sensitive_var_ids = sensitive_var_ids
    self.c_nb = NaiveBayes(root_params, leaf_params)

  def get_divergent_patterns(self, threshold, num_patterns, return_any=False):
    cdef Search dpf = Search(self.c_nb, self.target_value, threshold, self.sensitive_var_ids)
    cdef vector[DivergentPattern] patterns = dpf.getDivergentPatterns(num_patterns, return_any)
    cdef size_t num_visited = dpf.getNumNodes()
    return (patterns, num_visited)

  def get_discriminating_patterns(self, threshold, num_patterns, return_any=False):
    cdef Search dpf = Search(self.c_nb, self.target_value, threshold, self.sensitive_var_ids)
    cdef vector[DivergentPattern] patterns = dpf.getDiscriminatingPatterns(num_patterns, return_any)
    cdef size_t num_visited = dpf.getNumNodes()
    return (patterns, num_visited)    
