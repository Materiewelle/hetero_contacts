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

    steady_state s1({0,0.3,0.0});
    s1.solve<false>();
    steady_state s2({0,0,0.5});
    s2.solve<false>();
    steady_state s_final({0,0.3,0.5});
    s_final.solve<false>();
    steady_state s({0, 0.36, -1.0});
    s.solve();
    plot_ldos(s.phi, 1000);

    s = steady_state({0, 0.36, 1.0});
    s.solve();
    plot_ldos(s.phi, 1000);

    vec V_d, I;
    steady_state::output({0, 0.36, -1.0}, 1.0, 4000, V_d, I);

    int slope_begin = 20;
    int slope_len = 30;

    time_evolution te1;
    time_evolution te2;
    std::fill(begin(te1.V), begin(te1.V) + slope_begin - 1, voltage{0, .3, 0});
    std::fill(begin(te2.V), begin(te2.V) + slope_begin - 1, voltage{0, 0, .5});
    vec slope1 = linspace(0, 0.5, slope_len);
    vec slope2 = linspace(0, 0.3, slope_len);
    for (int i = 0; i < slope_len; ++i) {
        te1.V[slope_begin+i].d = slope1(i);
        te2.V[slope_begin+i].g = slope2(i);
    }
    std::fill(begin(te1.V) + slope_begin + slope_len, end(te1.V), voltage{0, .3, .5});
    std::fill(begin(te2.V) + slope_begin + slope_len, end(te2.V), voltage{0, .3, .5});

    te1.solve();
    te2.solve();

    vec I_t1(t::N_t);
    vec I_t2(t::N_t);
    for (int i = 0; i < t::N_t; ++i) {
        I_t1(i) = te1.I[i].total(d::N_x-1);
        I_t2(i) = te2.I[i].total(d::N_x-1);
    }

    vec I_sfinal(t::N_t);
    I_sfinal.fill(s_final.I.total(0));

    gnuplot gp1;
    gp1 << "set title \"transition Vd=0 -> Vd=0.5V\"\n";
    gp1 << "set terminal pdf rounded color enhanced font 'arial,12'\n";
    gp1 << "set output 'transition_drain.pdf'\n";
    vec I_s1(t::N_t);
    I_s1.fill(s1.I.total(0));
    gp1.add(I_s1);
    gp1.add(I_sfinal);
    gp1.add(I_t1);
    gp1.plot();

    gnuplot gp2;
    gp2 << "set title \"transition Vg=0 -> Vg=0.3V\"\n";
    gp2 << "set terminal pdf rounded color enhanced font 'arial,12'\n";
    gp2 << "set output 'transition_gate.pdf'\n";
    vec I_s2(t::N_t);
    I_s2.fill(s2.I.total(0));
    gp2.add(I_s2);
    gp2.add(I_sfinal);
    gp2.add(I_t2);
    gp2.plot();


//    plot(te.I[0].total);
//    plot(te.I[1].total);

//    cout << "sslv " << te.I[0].lv(0) << endl;
//    cout << "ssrv " << te.I[0].rv(0) << endl;
//    cout << "sslc " << te.I[0].lc(0) << endl;
//    cout << "ssrc " << te.I[0].rc(0) << endl;
//    cout << "sslt " << te.I[0].lt(0) << endl;
//    cout << "ssrt " << te.I[0].rt(0) << endl;

//    cout << "telv " << te.I[1].lv(0) << endl;
//    cout << "terv " << te.I[1].rv(0) << endl;
//    cout << "telc " << te.I[1].lc(0) << endl;
//    cout << "terc " << te.I[1].rc(0) << endl;
//    cout << "telt " << te.I[1].lt(0) << endl;
//    cout << "tert " << te.I[1].rt(0) << endl;


    return 0;
}
