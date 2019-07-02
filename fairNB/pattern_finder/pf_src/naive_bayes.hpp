#ifndef NAIVE_BAYES_HPP
#define NAIVE_BAYES_HPP

#include <vector>
#include <utility>

using std::pair;
using std::size_t;
using std::vector;

class NaiveBayes
{
public:
  NaiveBayes(){}
  NaiveBayes(const pair<double, double> &rootParams_,
                       const vector<vector<double>> &leafParams_)
    : rootParams(rootParams_), leafParams(leafParams_),
      numLeaves(leafParams_.size()) {}

  pair<double, double> rootParams;
  vector<vector<double>> leafParams;
  size_t numLeaves;
};

#endif
