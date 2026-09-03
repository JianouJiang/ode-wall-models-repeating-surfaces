// Model-side response of the deployed total-gradient TBLE wall model.
//
// Reads one face per line:  u_m dpds y_m nu tau_previous
// and re-solves the production kernel at u_m scaled by a set of factors, so
// that the logarithmic sensitivity
//
//     s = d ln|tau_w| / d ln u_m
//
// can be formed by central differences on the operator that actually ran.
// The kernel header is the one the deposited campaign compiled; nothing here
// re-implements it.
//
// Output, one line per face:
//   ok tau_0 tau_minus tau_plus roots_0 roots_minus roots_plus
// with ok = 1 only when all three solves converged on a unique branch.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "rswm_m13_tbleShoot_degenerate.H"

namespace
{

struct Solve
{
    double tau;
    int roots;
    bool ok;
};

Solve solveAt(double u, double dpds, double yM, double nu)
{
    // Cold, full-census solve: no continuation state is assumed, so the
    // response is a property of the model, not of the run's history.
    const TbleContinuationReport r =
        shootTauWContinuation(u, dpds, yM, nu, 0.41, 26.0, 0.0, false);
    Solve s;
    s.tau = r.tauW;
    s.roots = r.rootCount;
    s.ok = r.converged && !r.branchLoss && !r.ambiguous && !r.truncated
        && r.finite && std::isfinite(r.tauW);
    return s;
}

}  // namespace

int main(int argc, char** argv)
{
    double h = 0.02;
    if (argc > 1)
    {
        h = std::atof(argv[1]);
    }

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        double u = 0.0, dpds = 0.0, yM = 0.0, nu = 0.0, prev = 0.0;
        if (std::sscanf(line.c_str(), "%lf %lf %lf %lf %lf",
                        &u, &dpds, &yM, &nu, &prev) < 4)
        {
            std::printf("0 nan nan nan 0 0 0\n");
            continue;
        }

        const Solve s0 = solveAt(u, dpds, yM, nu);
        const Solve sm = solveAt(u*(1.0 - h), dpds, yM, nu);
        const Solve sp = solveAt(u*(1.0 + h), dpds, yM, nu);
        const int ok = (s0.ok && sm.ok && sp.ok) ? 1 : 0;
        std::printf("%d %.17g %.17g %.17g %d %d %d\n",
                    ok, s0.tau, sm.tau, sp.tau,
                    s0.roots, sm.roots, sp.roots);
    }
    return 0;
}
