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

//    steady_state s({0,0.7,0.8});
//    s.solve<false>();

//    plot_ldos(s.phi, 1000);

    time_evolution te;
    std::fill(begin(te.V), end(te.V), voltage{0, 0, 0});
    te.solve();


    return 0;
}
