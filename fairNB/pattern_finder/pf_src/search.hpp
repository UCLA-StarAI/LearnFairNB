#ifndef SEARCH_HPP
#define SEARCH_HPP

#include <cmath>
#include <utility>
#include <queue>
#include <vector>
#include "naive_bayes.hpp"

using std::make_pair;
using std::pair;
using std::priority_queue;
using std::size_t;
using std::vector;

typedef struct
{
  vector<pair<size_t, size_t>> base;
  vector<pair<size_t, size_t>> sens;

  double pDY, pNotDY, pDXY, pNotDXY;
  double kld;
} DivergentPattern;

typedef struct
{
  // P(d,features) and P(~d,features)
  double pDMax, pNotDMax; // instantiate remaining features to maximize P(d|features)
  double pDMin, pNotDMin; // minimize P(d|features)
} Extension;

class PatternComparator
{
public:
  int operator()(const DivergentPattern &p1, const DivergentPattern &p2)
  {
    return p1.kld > p2.kld;
  }
};

class Search
{
public:
  Search();
  Search(const NaiveBayes &NB_, size_t targetValue_, double threshold_,
         const vector<size_t> &sensitiveVarIDs_);

  // If stopAfterK=true, return any numPatterns number of discrimination patterns
  vector<DivergentPattern> getDivergentPatterns(size_t numPatterns, bool stopAfterK=false);
  vector<DivergentPattern> getDiscriminatingPatterns(size_t numPatterns, bool stopAfterK=false);
  size_t getNumNodes();

private:
  NaiveBayes NB;
  size_t targetValue;
  size_t numLeaves;
  size_t numS;
  double threshold;
  vector<bool> isSensitive;

  bool useKLD;
  bool stopAfterK;
  size_t numVisits;

  // Search states
  DivergentPattern curPattern;
  Extension baseExtension;     // keep current y, extend all other features
  Extension allExtension;      // keep current xy, extend all other features
  Extension baseOnlyExtension; // keep current y, extend features except already instantiated sensitive features
  Extension sensOnlyExtension; // keep current x, extend sensitive attributes
  vector<size_t> maxInstances;
  priority_queue<DivergentPattern, vector<DivergentPattern>, PatternComparator> patterns;

  void initializeSearch(size_t);
  void recurse(size_t);

  void addAssignmentToPattern(size_t, size_t, bool, bool = false);
  void removeAssignmentFromPattern(size_t, size_t, bool, bool = false);
  double computeDivergence();
  double computeDivergence2();
  double computeDivergenceBound();
  double computeDivergenceBound2();
  double computeDivergenceBoundHelper(double);
  double computeDifference();
  double computeDifferenceBound();
  double computeDifferenceBoundFor(double, double, double, double, bool);
  pair<double, double> computeDifferenceBound2();
};

#endif
