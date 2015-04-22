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
    potential phi = potential({0, 0.3, 0.6});
//    plot_ldos(phi, 1000);

    vec E = linspace(-20, 20, 500);

    cx_vec alt_s(500);
    cx_vec alt_d(500);
    cx_vec test_s(500);
    cx_vec test_d(500);
    for (int i = 0; i < 500; ++i) {
        self_energy(phi, E(i), alt_s(i), alt_d(i));
        self_energy_test(phi, E(i), test_s(i), test_d(i));
    }

    gnuplot gpsa;
    gpsa.add(alt_s);
    gpsa << "set title \"alt source\"\n";

    gnuplot gpst;
    gpst.add(test_s);
    gpst << "set title \"test source\"\n";

    gnuplot gpda;
    gpda.add(alt_d);
    gpda << "set title \"alt drain\"\n";

    gnuplot gpdt;
    gpdt.add(test_d);
    gpdt << "set title \"test drain\"\n";




    return 0;
}
