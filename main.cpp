//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>
#include "time_evolution.hpp"
#include "green.hpp"

using namespace arma;
using namespace std;

int main() {

    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

//    steady_state s({0, 0.4, 0.4});
//    s.solve();
//    cout << "sssstroeeem:" << s.I.total(0) << endl << endl;
//    plot_ldos(s.phi, 1000);


//    const int n = 30;
//    const double start = .2;
//    const double end = 1.;
//    const int parts = 10;
//    const double partsize = (end - start) / parts;

//    vec V_g, I;
//    for (int i = 0; i < parts; ++i) {
//        steady_state::transfer({0., start + i * partsize, .4}, start + (i + 1) * partsize, n, V_g, I);
//        mat data = join_horiz(V_g, I);
//        data.save("../../../../Fitting/data" + to_string(i), csv_ascii);
//    }

    double lam_min = .5;
    double lam_max = 4.;
    int steps = 10;
    double step = (lam_max - lam_min) / steps;

    vec V_g, I;
    for (int i = 0; i < steps; ++i) {
        double lam = lam_min + i * step;
        d::lam_d = lam;
        d::lam_g = lam;
        d::lam_s = lam;
        steady_state::transfer({0., .3, .4}, 1., 10, V_g, I);
        mat data = join_horiz(V_g, I);
        data.save("../../../../Fitting/lam" + to_string(i), csv_ascii);
    }

    return 0;
}
