
/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-15
 *  License: MIT
 */

#pragma once

void CUSUM(std::vector<float>& data, double stepsize, double h, double stddev)
{
    std::vector<float> cpos(data.size(), 0.0);
    std::vector<float> cneg(data.size(), 0.0);
    std::vector<float> gpos(data.size(), 0.0);
    std::vector<float> gneg(data.size(), 0.0);
}

void cumSum(double* x, int n, int s, int e, double stepsize, double h, double stdDev,
    std::vector<int>& v1, std::vector<double>& v2)
{
    int size = e - s + 1;
    double logp, logn = 0;
    double* cpos = new double[size];
    double* cneg = new double[size];
    double* gpos = new double[size];
    double* gneg = new double[size];
    for (int i = s; i <= e; i++)
    {
        cpos[i - s] = 0;
        cneg[i - s] = 0;
        gpos[i - s] = 0;
        gneg[i - s] = 0;
    }
    double anchor = s;
    double mean = x[s];
    double variance = stdDev * stdDev;
    int nstates = 0;
    double varM = x[s];
    double varS = 0;
    int jump = s;
    v1.push_back(s);
    for (int i = s + 1; i <= e; i++)
    {
        double varoldM = varM; // calculate move variance and mean.from mosaic.
        varM = varM + (x[i] - varM) / (i - anchor + 1);
        varS = varS + (x[i] - varoldM) * (x[i] - varM);
        variance = varS / (i - anchor + 1);
        mean = ((i - anchor) * mean + x[i]) / (i - anchor + 1);
        variance = (variance == 0) ? stdDev * stdDev : variance;

        logp = stepsize * stdDev / variance * (x[i] - mean - stepsize * stdDev / 2);
        logn = -1 * stepsize * stdDev / variance * (x[i] - mean + stepsize * stdDev / 2);
        cpos[i - s] = cpos[i - s - 1] + logp;
        cneg[i - s] = cneg[i - s - 1] + logn;
        gpos[i - s] = std::max(gpos[i - s - 1] + logp, 0.);
        gneg[i - s] = std::max(gneg[i - s - 1] + logn, 0.);
        if (gpos[i - s] >= h)
        {
            jump = s + argmin(cpos, anchor - s, i - s);
            v2.push_back(meancurrent(x, n, v1.back(), jump));
            v1.push_back(jump);
            nstates++;
            anchor = i;
            cpos[i - s] = 0;
            cneg[i - s] = 0;
            gpos[i - s] = 0;
            gneg[i - s] = 0;
            mean = x[i];
            varM = x[i];
        }
        else if (gneg[i - s] >= h)
        {
            jump = s + argmin(cneg, anchor - s, i - s);
            v2.push_back(meancurrent(x, n, v1.back(), jump));
            v1.push_back(jump);
            nstates++;
            anchor = i;
            cpos[i - s] = 0;
            cneg[i - s] = 0;
            gpos[i - s] = 0;
            gneg[i - s] = 0;
            mean = x[i];
            varM = x[i];
        }
    }

    v2.push_back(meancurrent(x, n, v1.back(), e));
    v1.push_back(e);

    delete[] cpos;
    delete[] cneg;
    delete[] gpos;
    delete[] gneg;
    return;
}

int argmin(double* x, int s, int e)
{
    double minval = x[s];
    int k = s;
    for (int i = s + 1; i <= e; i++)
    {
        if (x[i] < minval)
        {
            minval = x[i];
            k = i;
        }
    }
    return k;
}