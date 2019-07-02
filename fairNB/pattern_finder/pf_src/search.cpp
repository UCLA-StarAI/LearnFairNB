#include "search.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

using std::cout;
using std::endl;
using std::vector;
using std::numeric_limits;
using std::min;
using std::max;

namespace
{
bool leq(double a, double b)
{
    return a - 1E-12 <= b;
}

bool eq(double a, double b)
{
    return leq(a, b) && leq(b, a);
}

bool lessThan(double a, double b)
{
    return leq(a, b) && !eq(a, b);
}

bool is_nan(double x)
{
    return x != x;
}

} // namespace

Search::Search() {}

Search::Search(const NaiveBayes &NB_, size_t targetValue_, double threshold_,
               const vector<size_t> &sensitiveVarIDs_)
    : NB(NB_), targetValue(targetValue_), numLeaves(NB_.numLeaves), threshold(threshold_)
{
    isSensitive.assign(numLeaves, false);
    for (auto ID : sensitiveVarIDs_)
        isSensitive[ID] = true;
    numS = sensitiveVarIDs_.size();
}

vector<DivergentPattern> Search::getDivergentPatterns(size_t numPatterns, bool stopAfterK)
{
    this->useKLD = true;
    this->stopAfterK = stopAfterK;
    initializeSearch(numPatterns);
    recurse(0);

    cout << "Visited " << numVisits << " nodes" << endl;

    vector<DivergentPattern> patternVec;
    while (!patterns.empty())
    {
        DivergentPattern curPattern = patterns.top();
        if (lessThan(0,curPattern.kld)) patternVec.push_back(curPattern);
        patterns.pop();
    }
    return patternVec;
}

vector<DivergentPattern> Search::getDiscriminatingPatterns(size_t numPatterns, bool stopAfterK)
{
    this->useKLD = false;
    this->stopAfterK = stopAfterK;
    initializeSearch(numPatterns);
    recurse(0);

    cout << "Visited " << numVisits << " nodes" << endl;

    vector<DivergentPattern> patternVec;
    while (!patterns.empty())
    {
        DivergentPattern curPattern = patterns.top();
        if (lessThan(threshold, curPattern.kld)) patternVec.push_back(curPattern);
        patterns.pop();
    }
    return patternVec;
}

void Search::initializeSearch(size_t numPatterns)
{
    numVisits = 0;

    double decisionPrior[2] = {NB.rootParams.first, NB.rootParams.second};

    // Initial state (X = Y = empty)
    curPattern.base.clear();
    curPattern.sens.clear();
    curPattern.base.reserve(numLeaves);
    curPattern.sens.reserve(numS);
    curPattern.pDY = curPattern.pDXY = decisionPrior[targetValue];
    curPattern.pNotDY = curPattern.pNotDXY = decisionPrior[1 - targetValue];

    maxInstances.clear();
    maxInstances.reserve(numLeaves);
    allExtension.pDMax = allExtension.pDMin = decisionPrior[targetValue];
    allExtension.pNotDMax = allExtension.pNotDMin = decisionPrior[1 - targetValue];
    sensOnlyExtension.pDMax = sensOnlyExtension.pDMin = decisionPrior[targetValue];
    sensOnlyExtension.pNotDMax = sensOnlyExtension.pNotDMin = decisionPrior[1 - targetValue];
    for (size_t i = 0; i < numLeaves; i++)
    {
        double params[2][2] = {
            {NB.leafParams[i][0], NB.leafParams[i][2]},
            {NB.leafParams[i][1], NB.leafParams[i][3]}};

        size_t maxI =
            (params[0][targetValue] / params[0][1 - targetValue] > params[1][targetValue] / params[1][1 - targetValue])
                ? 0
                : 1;
        size_t minI = 1 - maxI;
        maxInstances[i] = maxI;

        allExtension.pDMax *= params[maxI][targetValue];
        allExtension.pNotDMax *= params[maxI][1 - targetValue];
        allExtension.pDMin *= params[minI][targetValue];
        allExtension.pNotDMin *= params[minI][1 - targetValue];

        if (isSensitive[i])
        {
            sensOnlyExtension.pDMax *= params[maxI][targetValue];
            sensOnlyExtension.pNotDMax *= params[maxI][1 - targetValue];
            sensOnlyExtension.pDMin *= params[minI][targetValue];
            sensOnlyExtension.pNotDMin *= params[minI][1 - targetValue];
        }
    }
    baseOnlyExtension.pDMax = baseExtension.pDMax = allExtension.pDMax;
    baseOnlyExtension.pNotDMax = baseExtension.pNotDMax = allExtension.pNotDMax;
    baseOnlyExtension.pDMin = baseExtension.pDMin = allExtension.pDMin;
    baseOnlyExtension.pNotDMin = baseExtension.pNotDMin = allExtension.pNotDMin;

    // Initialize min-heap
    patterns = priority_queue<DivergentPattern, vector<DivergentPattern>, PatternComparator>();
    for (size_t i = 0; i < numPatterns; i++)
    {
        DivergentPattern p;
        p.pDY = decisionPrior[targetValue];
        p.pNotDY = decisionPrior[1-targetValue];
        p.pDXY = decisionPrior[targetValue];
        p.pNotDXY = decisionPrior[1-targetValue];
        p.kld = useKLD ? 0 : threshold;
        patterns.push(p);
    }
}

// optimize KLD heuristic if useKLD is true, otherwise optimize discrimination score
void Search::recurse(size_t level)
{
    if (level >= numLeaves
        || (stopAfterK && patterns.top().kld != (useKLD ? 0 : threshold)))
        return;
    numVisits++;

    // If sensitive attribute, add to sens and recurse
    if (isSensitive[level])
    {
        // Compute Pr(d | y)
        for (size_t val = 0; val <= 1; val++)
        {
            addAssignmentToPattern(level, val, true);

            curPattern.kld = useKLD ? computeDivergence2() : computeDifference();
            if (leq(patterns.top().kld, curPattern.kld))
            {
                patterns.pop();
                patterns.push(curPattern);
            }
            if (level + 1 < numLeaves)
            {
                double bound = useKLD ? computeDivergenceBound2() : computeDifferenceBound();
                // cout << "Divergence: " << curPattern.kld << "; Bound: " << bound << endl;

                if (bound > patterns.top().kld)
                {
                    recurse(level + 1);
                }
            }

            removeAssignmentFromPattern(level, val, true);
        }
    }

    // Add to base and recurse
    for (size_t val = 0; val <= 1; val++)
    {
        addAssignmentToPattern(level, val, false);

        curPattern.kld = useKLD ? computeDivergence2() : computeDifference();
        if (leq(patterns.top().kld, curPattern.kld))
        {
            patterns.pop();
            patterns.push(curPattern);
        }

        if (level + 1 < numLeaves)
        {
            double bound = useKLD ? computeDivergenceBound2() : computeDifferenceBound();
            // cout << "Divergence: " << curPattern.kld << "; Bound: " << bound << endl;

            if (bound > patterns.top().kld)
            {
                recurse(level + 1);
            }
        }

        removeAssignmentFromPattern(level, val, false);
    }

    // Skip and recurse
    addAssignmentToPattern(level, -1, false, true);
    recurse(level + 1);
    removeAssignmentFromPattern(level, -1, false, true);
}

// TODO: make extensions a class with skip&add function
void Search::addAssignmentToPattern(size_t var, size_t val, bool sens, bool skip)
{
    double maxParams[2] = {
        NB.leafParams[var][maxInstances[var]],
        NB.leafParams[var][maxInstances[var] + 2]};
    double minParams[2] = {
        NB.leafParams[var][(1 - maxInstances[var])],
        NB.leafParams[var][(1 - maxInstances[var]) + 2]};

    if (skip)
    {
        // Do nothing to curPattern
        // Remove var from extensions
        allExtension.pDMax /= maxParams[targetValue];
        allExtension.pNotDMax /= maxParams[1 - targetValue];
        allExtension.pDMin /= minParams[targetValue];
        allExtension.pNotDMin /= minParams[1 - targetValue];

        baseExtension.pDMax /= maxParams[targetValue];
        baseExtension.pNotDMax /= maxParams[1 - targetValue];
        baseExtension.pDMin /= minParams[targetValue];
        baseExtension.pNotDMin /= minParams[1 - targetValue];

        baseOnlyExtension.pDMax /= maxParams[targetValue];
        baseOnlyExtension.pNotDMax /= maxParams[1 - targetValue];
        baseOnlyExtension.pDMin /= minParams[targetValue];
        baseOnlyExtension.pNotDMin /= minParams[1 - targetValue];

        if (isSensitive[var])
        {
            sensOnlyExtension.pDMax /= maxParams[targetValue];
            sensOnlyExtension.pNotDMax /= maxParams[1 - targetValue];
            sensOnlyExtension.pDMin /= minParams[targetValue];
            sensOnlyExtension.pNotDMin /= minParams[1 - targetValue];
        }
        return;
    }

    double *toAdd = (val == maxInstances[var]) ? maxParams : minParams;
    double *toRemove = (val == maxInstances[var]) ? minParams : maxParams;

    curPattern.pDXY *= toAdd[targetValue];
    curPattern.pNotDXY *= toAdd[1 - targetValue];

    pair<size_t, size_t> assignment = make_pair(var, val);
    if (sens)
    {
        curPattern.sens.push_back(assignment);
    }
    else
    {
        curPattern.base.push_back(assignment);
        curPattern.pDY *= toAdd[targetValue];
        curPattern.pNotDY *= toAdd[1 - targetValue];
    }

    double dRatio = toAdd[targetValue] / toRemove[targetValue];
    double notDRatio = toAdd[1 - targetValue] / toRemove[1 - targetValue];
    if (val != maxInstances[var])
    {
        allExtension.pDMax *= dRatio;
        allExtension.pNotDMax *= notDRatio;
        if (sens)
        {
            sensOnlyExtension.pDMax *= dRatio;
            sensOnlyExtension.pNotDMax *= notDRatio;
        }
        else
        {
            baseExtension.pDMax *= dRatio;
            baseExtension.pNotDMax *= notDRatio;
            baseOnlyExtension.pDMax *= dRatio;
            baseOnlyExtension.pNotDMax *= notDRatio;
        }
    }
    else
    {
        allExtension.pDMin *= dRatio;
        allExtension.pNotDMin *= notDRatio;
        if (sens)
        {
            sensOnlyExtension.pDMin *= dRatio;
            sensOnlyExtension.pNotDMin *= notDRatio;
        }
        else
        {
            baseExtension.pDMin *= dRatio;
            baseExtension.pNotDMin *= notDRatio;
            baseOnlyExtension.pDMin *= dRatio;
            baseOnlyExtension.pNotDMin *= notDRatio;
        }
    }

    if (sens)
    {
        baseOnlyExtension.pDMax /= maxParams[targetValue];
        baseOnlyExtension.pNotDMax /= maxParams[1 - targetValue];
        baseOnlyExtension.pDMin /= minParams[targetValue];
        baseOnlyExtension.pNotDMin /= minParams[1 - targetValue];
    }
    else if (isSensitive[var])
    {
        sensOnlyExtension.pDMax /= maxParams[targetValue];
        sensOnlyExtension.pNotDMax /= maxParams[1 - targetValue];
        sensOnlyExtension.pDMin /= minParams[targetValue];
        sensOnlyExtension.pNotDMin /= minParams[1 - targetValue];
    }
}

// Pop the last assignment
void Search::removeAssignmentFromPattern(size_t var, size_t val, bool sens, bool skip)
{
    double maxParams[2] = {
        NB.leafParams[var][maxInstances[var]],
        NB.leafParams[var][maxInstances[var] + 2]};
    double minParams[2] = {
        NB.leafParams[var][(1 - maxInstances[var])],
        NB.leafParams[var][(1 - maxInstances[var]) + 2]};

    if (skip)
    {
        // Do nothing to curPattern
        // Remove var from extensions
        allExtension.pDMax *= maxParams[targetValue];
        allExtension.pNotDMax *= maxParams[1 - targetValue];
        allExtension.pDMin *= minParams[targetValue];
        allExtension.pNotDMin *= minParams[1 - targetValue];

        baseExtension.pDMax *= maxParams[targetValue];
        baseExtension.pNotDMax *= maxParams[1 - targetValue];
        baseExtension.pDMin *= minParams[targetValue];
        baseExtension.pNotDMin *= minParams[1 - targetValue];

        baseOnlyExtension.pDMax *= maxParams[targetValue];
        baseOnlyExtension.pNotDMax *= maxParams[1 - targetValue];
        baseOnlyExtension.pDMin *= minParams[targetValue];
        baseOnlyExtension.pNotDMin *= minParams[1 - targetValue];

        if (isSensitive[var])
        {
            sensOnlyExtension.pDMax *= maxParams[targetValue];
            sensOnlyExtension.pNotDMax *= maxParams[1 - targetValue];
            sensOnlyExtension.pDMin *= minParams[targetValue];
            sensOnlyExtension.pNotDMin *= minParams[1 - targetValue];
        }
        return;
    }

    double *toRemove = (val == maxInstances[var]) ? maxParams : minParams;
    double *toAdd = (val == maxInstances[var]) ? minParams : maxParams;

    curPattern.pDXY /= toRemove[targetValue];
    curPattern.pNotDXY /= toRemove[1 - targetValue];

    if (sens)
    {
        curPattern.sens.pop_back();
    }
    else
    {
        curPattern.pDY /= toRemove[targetValue];
        curPattern.pNotDY /= toRemove[1 - targetValue];
        curPattern.base.pop_back();
    }

    double dRatio = toAdd[targetValue] / toRemove[targetValue];
    double notDRatio = toAdd[1 - targetValue] / toRemove[1 - targetValue];
    if (val != maxInstances[var])
    {
        allExtension.pDMax *= dRatio;
        allExtension.pNotDMax *= notDRatio;
        if (sens)
        {
            sensOnlyExtension.pDMax *= dRatio;
            sensOnlyExtension.pNotDMax *= notDRatio;
        }
        else
        {
            baseExtension.pDMax *= dRatio;
            baseExtension.pNotDMax *= notDRatio;
            baseOnlyExtension.pDMax *= dRatio;
            baseOnlyExtension.pNotDMax *= notDRatio;
        }
    }
    else
    {
        allExtension.pDMin *= dRatio;
        allExtension.pNotDMin *= notDRatio;
        if (sens)
        {
            sensOnlyExtension.pDMin *= dRatio;
            sensOnlyExtension.pNotDMin *= notDRatio;
        }
        else
        {
            baseExtension.pDMin *= dRatio;
            baseExtension.pNotDMin *= notDRatio;
            baseOnlyExtension.pDMin *= dRatio;
            baseOnlyExtension.pNotDMin *= notDRatio;
        }
    }

    if (sens)
    {
        baseOnlyExtension.pDMax *= maxParams[targetValue];
        baseOnlyExtension.pNotDMax *= maxParams[1 - targetValue];
        baseOnlyExtension.pDMin *= minParams[targetValue];
        baseOnlyExtension.pNotDMin *= minParams[1 - targetValue];
    }
    else if (isSensitive[var])
    {
        sensOnlyExtension.pDMax *= maxParams[targetValue];
        sensOnlyExtension.pNotDMax *= maxParams[1 - targetValue];
        sensOnlyExtension.pDMin *= minParams[targetValue];
        sensOnlyExtension.pNotDMin *= minParams[1 - targetValue];
    }
}

double Search::computeDivergence()
{
    double pXY = curPattern.pDXY + curPattern.pNotDXY;
    double pY = curPattern.pDY + curPattern.pNotDY;
    double delta = (curPattern.pDXY / pXY) - (curPattern.pDY / pY);
    if (fabs(delta) <= threshold)
    {
        return 0;
    }

    // Odds of X given Y
    double sensOdds = pXY / (pY - pXY);
    double q = (delta < 0) ? sensOdds * (curPattern.pDY - curPattern.pDXY - (pY * threshold))
                           : sensOdds * (curPattern.pDY - curPattern.pDXY + (pY * threshold));

    // Problem is not feasible (should never be here)
    if (q <= 0 || q > pXY)
    {
        return numeric_limits<double>::infinity();
    }

    double score = (curPattern.pDXY * (log2(curPattern.pDXY) - log2(q))) + (curPattern.pNotDXY * (log2(curPattern.pNotDXY) - log2(pXY - q)));

    return score;
}

double Search::computeDivergence2()
{
    double pXY = curPattern.pDXY + curPattern.pNotDXY;
    double pY = curPattern.pDY + curPattern.pNotDY;
    double delta = (curPattern.pDXY / pXY) - (curPattern.pDY / pY);

    if (fabs(delta) <= threshold)
    {
        return 0;
    }

    double coeff = 1 / (1 / pXY - 1 / pY);
    double q = (delta < 0) ? coeff * (-threshold - delta) : coeff * (threshold - delta);

    double score = (curPattern.pDXY * (log2(curPattern.pDXY) - log2(q + curPattern.pDXY))) + (curPattern.pNotDXY * (log2(curPattern.pNotDXY) - log2(curPattern.pNotDXY - q)));
    return score;
}

double Search::computeDivergenceBound()
{
    double pDAll = allExtension.pDMax / (allExtension.pDMax + allExtension.pNotDMax);
    double pNotDAll = allExtension.pNotDMin / (allExtension.pDMin + allExtension.pNotDMin);
    double pDBase = baseExtension.pDMin / (baseExtension.pDMin + baseExtension.pNotDMin);
    double pNotDBase = baseExtension.pNotDMax / (baseExtension.pDMax + baseExtension.pNotDMax);
    return curPattern.pDXY * log2(pDAll / pDBase) + curPattern.pNotDXY * log2(pNotDAll / pNotDBase);
}

double Search::computeDivergenceBoundHelper(double diff)
{
    // largest P(x | y) possible by extending
    double pxy1 = (allExtension.pDMax + allExtension.pNotDMax) / (baseOnlyExtension.pDMax + baseOnlyExtension.pNotDMax);
    double pxy2 = (allExtension.pDMin + allExtension.pNotDMin) / (baseOnlyExtension.pDMin + baseOnlyExtension.pNotDMin);
    double maxPx_y = max(pxy1, pxy2);
    double minPx_y = min(pxy1, pxy2);
        // (allExtension.pDMax + allExtension.pNotDMax) / (baseOnlyExtension.pDMax + baseOnlyExtension.pNotDMax),
        // (allExtension.pDMin + allExtension.pNotDMin) / (baseOnlyExtension.pDMin + baseOnlyExtension.pNotDMin));

    if (leq(0, diff))
    { // diff is positive
        double pNotDXY_min = allExtension.pNotDMin / (allExtension.pDMin + allExtension.pNotDMin);
        double pNotDXY_max = allExtension.pNotDMax / (allExtension.pDMax + allExtension.pNotDMax);
        double try_bound = curPattern.pNotDXY * log2(pNotDXY_min * (1 - minPx_y) / ((pNotDXY_max * (1 - maxPx_y)) - diff));
        // double try_bound = -curPattern.pNotDXY * log2(1 - diff / (pNotDXY * (1 - maxPx_y)));
        if (is_nan(try_bound)) return numeric_limits<double>::infinity();
        return max(try_bound, 0.0);
    }
    else
    {
        double pDXY_max = allExtension.pDMax / (allExtension.pDMax + allExtension.pNotDMax);
        double pDXY_min = allExtension.pDMin / (allExtension.pDMin + allExtension.pNotDMin);
        double try_bound = curPattern.pDXY * log2(pDXY_max * (1 - minPx_y) / ((pDXY_min * (1 - maxPx_y)) - diff));
        // double try_bound = -curPattern.pDXY * log2(1 + diff / (pDXY * (1 - maxPx_y)));
        if (is_nan(try_bound)) return numeric_limits<double>::infinity();
        return max(try_bound, 0.0);
    }
}

double Search::computeDivergenceBound2()
{
    // double pXY = curPattern.pDXY + curPattern.pNotDXY;
    // double pY = curPattern.pDY + curPattern.pNotDY;
    // double delta = (curPattern.pDXY / pXY) - (curPattern.pDY / pY);

    pair<double, double> diffBounds = computeDifferenceBound2();
    if (leq(diffBounds.first, threshold) && leq(-threshold, diffBounds.second)) return 0.0;
    double score1 = computeDivergenceBoundHelper(threshold - diffBounds.first);
    double score2 = computeDivergenceBoundHelper(-threshold - diffBounds.second);

    // double coeff = 1 / (1/pXY - 1/pY);
    // double q1 = coeff * (threshold - diffBounds.first);
    // double q2 = coeff * (-threshold - diffBounds.second);
    // double score1 = (curPattern.pDXY * (log2(curPattern.pDXY) - log2(q1 + curPattern.pDXY)))
    //                + (curPattern.pNotDXY * (log2(curPattern.pNotDXY) - log2(curPattern.pNotDXY - q1)));
    // double score2 = (curPattern.pDXY * (log2(curPattern.pDXY) - log2(q2 + curPattern.pDXY)))
    //                + (curPattern.pNotDXY * (log2(curPattern.pNotDXY) - log2(curPattern.pNotDXY - q2)));

    double altBound = computeDivergenceBound();
    if (is_nan(score1) || is_nan(score2))
    {
        return altBound;
    }
    else
    {
        return min(altBound, max(score1, score2));
    }
}

double Search::computeDifference()
{
    double pXY = curPattern.pDXY + curPattern.pNotDXY;
    double pY = curPattern.pDY + curPattern.pNotDY;
    return fabs((curPattern.pDXY / pXY) - (curPattern.pDY / pY));
}

// a = Pr(x | d)
// b = Pr(x | ~d)
double Search::computeDifferenceBoundFor(double a, double b, double lower, double upper, bool maxi)
{
    if (eq(a, b))
        return 0.0;
    else if (eq(a, 0) || eq(b, 0))
        return 1.0;
    else
    {
        double opt = (b - sqrt(a * b)) / (b - a);
        double opt_value = a * opt / (a * opt + b * (1 - opt)) - opt;
        double l_value = a * lower / (a * lower + b * (1 - lower)) - lower;
        double u_value = a * upper / (a * upper + b * (1 - upper)) - upper;
        if (maxi)
            return max(opt_value, max(l_value, u_value));
        else
            return min(opt_value, min(l_value, u_value));
        /*
        if (leq(lower, opt) && leq(opt, upper))
        {
            return a * opt / (a * opt + b * (1 - opt)) - opt;
        }
        else if (leq(opt, lower))
        {
            return a * lower / (a * lower + b * (1 - lower)) - lower;
        }
        else
        {
            return a * upper / (a * upper + b * (1 - upper)) - upper;
        }
        */
    }
}

double Search::computeDifferenceBound()
{
    auto bounds = computeDifferenceBound2();
    return max(fabs(bounds.first), fabs(bounds.second));
}

pair<double, double> Search::computeDifferenceBound2()
{
    double decisionPrior[2] = {NB.rootParams.first, NB.rootParams.second};
    double a_max = sensOnlyExtension.pDMax / decisionPrior[targetValue];
    double b_max = sensOnlyExtension.pNotDMax / decisionPrior[1 - targetValue];
    double a_min = sensOnlyExtension.pDMin / decisionPrior[targetValue];
    double b_min = sensOnlyExtension.pNotDMin / decisionPrior[1 - targetValue];

    double lower = baseExtension.pDMin / (baseExtension.pDMin + baseExtension.pNotDMin);
    double upper = baseExtension.pDMax / (baseExtension.pDMax + baseExtension.pNotDMax);

    double u_bound = computeDifferenceBoundFor(a_max, b_max, lower, upper, true);
    double l_bound = computeDifferenceBoundFor(a_min, b_min, lower, upper, false);
    // if (eq(l_bound, u_bound))
    // {
    //     if (lessThan(0, l_bound))
    //         l_bound = 0;
    //     if (lessThan(u_bound, 0))
    //         u_bound = 0;
    // }

    return make_pair(u_bound, l_bound);
}

size_t Search::getNumNodes()
{
    return numVisits;
}

/*
int main() {
    // cout.width(12);
    // cout.precision(12);

    pair<double,double> rootParams = {0.8752918048900357, 0.12470819510996436};
    vector<vector<double>> leafParams = {{0.22873563218390805, 0.771264367816092, 0.33408197641774284, 0.6659180235822572},
                                          {0.05763546798029557, 0.9423645320197044, 0.14020681265206814, 0.8597931873479319},
                                          {0.8466338259441708, 0.15336617405582922, 0.7678270634475014, 0.2321729365524986},
                                          {0.5494252873563218, 0.45057471264367815, 0.3950963877971177, 0.6049036122028822},
                                          {0.558128078817734, 0.44187192118226604, 0.5097557551937114, 0.4902442448062886},
                                          {0.2471264367816092, 0.7528735632183908, 0.29664982219726743, 0.7033501778027326}, 
                                          {0.06321839080459771, 0.9367816091954023, 0.873385738349242, 0.126614261650758}};
    NaiveBayes NB(rootParams, leafParams);
    vector<size_t> sensIDs = {0,1,2,3};
    Search s(NB, 1, 0.01, sensIDs);
    
    // pair<double, double> rootParams = {0.3, 0.7};
    // vector<vector<double>> leafParams = {{0.3, 0.7, 0.1, 0.9},
    //                                      {0.4, 0.6, 0.5, 0.5},
    //                                      {0.2, 0.8, 0.3, 0.7},
    //                                      {0.15, 0.85, 0.25, 0.75}};
    // NaiveBayes NB(rootParams, leafParams);
    // vector<size_t> sensIDs = {0,2};
    // Search s(NB, 0, 0.01, sensIDs);
    
    // auto patterns = s.getDivergentPatterns(100000);
    // cout << "Number of patterns returned : " << patterns.size() << endl;
    // for (int i = 0; i < patterns.size(); i++) {
    //     cout << patterns[i].kld << endl;
    // }

    auto patterns = s.getDiscriminatingPatterns(100000);
    cout << "Number of patterns returned : " << patterns.size() << endl;
    for (int i = 0; i < patterns.size(); i++) {
        cout << patterns[i].kld << endl;
    }
}
*/