/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */

#include <fstream>
#include <vector>
#include "hmm_gmm.h"


void test() {
    std::string fn_data = "/home/yuanjie/Projects/capmod/blaR_cap0.bin";
    std::string fn_out = "/home/yuanjie/Projects/capmod/blaR_cap0_align.bin";
    std::string fn_model = "/home/yuanjie/Projects/capmod/blaR_cap0.txt";
    std::string fn_fa = "/home/yuanjie/Projects/capmod/DNA_tag_103.fa";
    
    StepFitHMMGMM model;
    model.loadData(fn_data, false);
    model.allocateModel(fn_fa, "T7-blaR", 3, 1);
    model.train(10, 1, 20, 2, 1.0, "et", 41);
    model.saveModel(fn_model);
    model.infer(10, 1, fn_out);
}

int main()
{
    test();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add
//   existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln
//   file
