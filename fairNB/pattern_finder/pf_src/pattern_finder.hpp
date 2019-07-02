#ifndef PATTERN_FINDER_HPP
#define PATTERN_FINDER_HPP

#include <vector>
#include <utility>
#include <cmath>
#include "naive_bayes.hpp"

using std::make_pair;
using std::pair;
using std::size_t;
using std::vector;

typedef struct
{
  vector<pair<size_t, size_t>> base;
  vector<pair<size_t, size_t>> sens;
  double pBase;
  double pAll;
  double pDX;
  double pD_X;
  double pX;
  double pDY;
  double pD_Y;
  double pY;
  double pDXY;
  double pD_XY;
  double pXY;
} Pattern;

class PatternFinder
{
public:
  PatternFinder();
  PatternFinder(const NaiveBayes &NB_, size_t targetValue_,
                const vector<size_t> &sensitiveVarIDs_);

  void findPatterns();
  vector<Pattern> getPatterns();

private:
  NaiveBayes NB;
  size_t targetValue;
  vector<size_t> sensitiveVarIDs;
  size_t numLeaves;
  size_t numN, numS;
  vector<bool> isSensitive;
  vector<pair<size_t, size_t>> base, sens;
  vector<Pattern> patterns;

  void includeAssignmentProbs(
      const vector<pair<size_t, size_t>> &assignments, 
      double *values) const;

  void computeProbabilities(const vector<pair<size_t, size_t>> &x,
                            const vector<pair<size_t, size_t>> &y,
                            Pattern &pattern) const;

  void recurse(size_t);
  void recordPattern();
};

#endif
