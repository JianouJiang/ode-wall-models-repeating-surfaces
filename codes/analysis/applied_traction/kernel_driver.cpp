/*---------------------------------------------------------------------------*\
  Independent driver for the deployed TBLE wall-model kernel.

  The production kernel header is included VERBATIM from the case's own
  `input/wallmodel_tble/` directory (path supplied at compile time through
  -DRSWM_KERNEL_HEADER).  Nothing in this file re-implements the model: it
  only feeds the kernel the per-face inputs that the solver logged and prints
  the resulting state at full precision, so that an external comparison can
  establish whether the deployed boundary condition is reproduced exactly.

  Input  (stdin, one record per line, whitespace separated):
      UMatch  dpds  yM  nu  kappa  Aplus  tangentialSpeed
  Output (stdout, one record per line):
      tauW rootCount selectedRoot homotopySteps converged branchLoss
      ambiguous truncated finite rawNut upperNut nut appliedTauS
      appliedTractionMagnitude lowerClipped vectorCapped projFinite
\*---------------------------------------------------------------------------*/
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>

#include RSWM_KERNEL_HEADER

int main()
{
    std::ios::sync_with_stdio(false);
    std::cout << std::setprecision(17);

    double UMatch, dpds, yM, nu, kappa, Aplus, speed;
    while (std::cin >> UMatch >> dpds >> yM >> nu >> kappa >> Aplus >> speed)
    {
        // Every logged face carries homotopySteps=33, i.e. the solver took the
        // first-solve branch with hasPrevious=false.  The driver reproduces
        // exactly that call; no continuation state is invented.
        const TbleContinuationReport report = shootTauWContinuation
        (
            UMatch, dpds, yM, nu, kappa, Aplus,
            0.0,    // previousTau, unused when hasPrevious is false
            false   // hasPrevious
        );

        const TbleNutProjection projection = tbleVectorRealizableNut
        (
            report.tauW, UMatch, speed, yM, nu
        );

        std::cout
            << report.tauW << ' '
            << report.rootCount << ' '
            << report.selectedRoot << ' '
            << report.homotopySteps << ' '
            << int(report.converged) << ' '
            << int(report.branchLoss) << ' '
            << int(report.ambiguous) << ' '
            << int(report.truncated) << ' '
            << int(report.finite) << ' '
            << projection.rawNut << ' '
            << projection.upperNut << ' '
            << projection.nut << ' '
            << projection.appliedTauS << ' '
            << projection.appliedTractionMagnitude << ' '
            << int(projection.lowerClipped) << ' '
            << int(projection.vectorCapped) << ' '
            << int(projection.finite) << '\n';
    }
    return 0;
}
