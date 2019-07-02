#include "pattern_finder.hpp"

PatternFinder::PatternFinder() {}

PatternFinder::PatternFinder(const NaiveBayes &NB_, size_t targetValue_,
                             const vector<size_t> &sensitiveVarIDs_)
    : NB(NB_), targetValue(targetValue_),
      sensitiveVarIDs(sensitiveVarIDs_),
      numLeaves(NB_.numLeaves)
{
    isSensitive.assign(numLeaves, false);
    for (auto ID : sensitiveVarIDs)
        isSensitive[ID] = true;
    numS = sensitiveVarIDs.size();
    numN = numLeaves - numS;

    base.reserve(numLeaves);
    sens.reserve(numS);
}

void PatternFinder::findPatterns(void)
{
    patterns.clear();
    size_t numPatterns = pow(3, numN) * pow(5, numS) - pow(3, numLeaves);
    patterns.reserve(numPatterns);
    base.clear();
    sens.clear();
    recurse(0);
}

void PatternFinder::recurse(size_t level)
{

    if (level == numLeaves)
    {
        if (!sens.empty())
            recordPattern();
        return;
    }

    // add to base and recurse
    for (size_t value = 0; value <= 1; value++)
    {
        pair<size_t, size_t> assignment = make_pair(level, value);

        base.push_back(assignment);
        recurse(level + 1);
        base.pop_back();
    }

    // if sensitive, add to sens and recurse
    if (isSensitive[level])
        for (size_t value = 0; value <= 1; value++)
        {
            pair<size_t, size_t> assignment = make_pair(level, value);
            sens.push_back(assignment);
            recurse(level + 1);
            sens.pop_back();
        }

    // skip and recurse
    recurse(level + 1);
}

void PatternFinder::recordPattern()
{
    Pattern p;
    p.base = base;
    p.sens = sens;
    computeProbabilities(base, sens, p);
    patterns.push_back(p);
}

void PatternFinder::includeAssignmentProbs(
    const vector<pair<size_t, size_t>> &assignments,
    double *value) const
{
    for (int d = 0; d <= 1; d++)
        for (auto vv : assignments)
            value[d] *= NB.leafParams[vv.first][2 * d + vv.second];
}

void PatternFinder::computeProbabilities(
    const vector<pair<size_t, size_t>> &y,
    const vector<pair<size_t, size_t>> &x,
    Pattern &pattern) const
{
    double value[2] = {NB.rootParams.first, NB.rootParams.second};

    includeAssignmentProbs(y, value);
    pattern.pDY = value[targetValue];
    pattern.pD_Y = value[1 - targetValue];
    pattern.pY = pattern.pDY + pattern.pD_Y;


    includeAssignmentProbs(x, value);
    pattern.pDXY = value[targetValue];
    pattern.pD_XY = value[1 - targetValue];
    pattern.pXY = pattern.pDXY + pattern.pD_XY;

    
    value[0] = NB.rootParams.first;
    value[1] = NB.rootParams.second;
    includeAssignmentProbs(x, value);
    pattern.pDX = value[targetValue];
    pattern.pD_X = value[1-targetValue];
    pattern.pX = pattern.pDX + pattern.pD_X;
    
    pattern.pAll = pattern.pDXY / pattern.pXY;
    pattern.pBase = pattern.pDY / pattern.pY;
}

vector<Pattern> PatternFinder::getPatterns()
{
    return patterns;
}
