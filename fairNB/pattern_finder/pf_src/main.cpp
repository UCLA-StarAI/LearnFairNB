#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

#include "naive_bayes.hpp"
#include "pattern_finder.hpp"

using std::cout;
using std::endl;
using std::ifstream;
using std::make_pair;
using std::pair;
using std::vector;

void readParamFile(char *filename,
                   pair<double, double> &rootParams,
                   vector<vector<double>> &leafParams)
{
    ifstream ifs;
    ifs.open(filename);

    size_t numLeaves;
    ifs >> numLeaves;
    leafParams.resize(numLeaves, vector<double>(4));
    ifs >> rootParams.first >> rootParams.second;

    for (size_t i = 0; i < numLeaves; i++)
        for (size_t j = 0; j < 4; j++)
            ifs >> leafParams[i][j];

    ifs.close();
}

void printFormula(const vector<pair<size_t, size_t>> &f)
{
    for (auto a : f)
        cout << a.first << "=" << a.second << " ";
    cout << endl;
}

int main(int, char **argv)
{
    pair<double, double> rootParams;
    vector<vector<double>> leafParams;
    readParamFile(argv[1], rootParams, leafParams);
    NaiveBayes ToyNB(rootParams, leafParams);

    vector<pair<size_t, size_t>> a = {{0, 1}, {1, 1}, {2, 1}};

    // P(y=0 | x0 = 1, x1 = 1, x2 = 1)
    cout << ToyNB.computeProbability(0, a) << endl;
    // P(y=0 | x0 = 1, x1 = 1, x2 = 1)
    cout << ToyNB.computeProbability(1, a) << endl;

    PatternFinder pf(ToyNB, 0, {1, 3});
    pf.findPatterns();
    vector<Pattern> patterns = pf.getPatterns();
    for (auto pattern : patterns)
    {
        cout << "base: ";
        printFormula(pattern.base);
        cout << "sens: ";
        printFormula(pattern.sens);
        cout << endl;
    }
}
