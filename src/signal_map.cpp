/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */

#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include "hmm_gmm.h"

void predict() {
    std::string fn_data = "/home/yuanjie/Projects/capmod/20260615_AR_cap_blaR.bin";
    std::string fn_json = "/home/yuanjie/Projects/4sU_nanopore/multichain.json";
    std::string fn_out = "/home/yuanjie/Projects/4sU_nanopore/multichain_train.json";

    StepFitHMMGMM model;
    model.loadModel(fn_json);
    model.segment(fn_data, 10, "",fn_out);
    //std::vector<std::string> uuids;
    //std::vector<Real> yreal;
    //std::vector<Real> ypredict;
    //std::ofstream fout("result.tsv");
    //if (!fout.is_open()) {
    //    throw std::runtime_error("Cannot open result.tsv");
    //}
    //fout << "uuid\tyreal\typredict\n";
    //size_t n = std::min({uuids.size(), yreal.size(), ypredict.size()});
    //for (size_t i = 0; i < n; ++i) {
    //    fout << uuids[i] << '\t'
    //         << std::setprecision(10) << yreal[i] << '\t'
    //         << std::setprecision(10) << ypredict[i] << '\n';
    //}
    //fout.close();
}

void train() {
    std::string fn_data = "/home/yuanjie/Projects/capmod/20260615_AR_cap_blaR.bin";
    std::string fn_out_data = "/home/yuanjie/Projects/capmod/20260615_AR_cap_blaR_hmm.bin";
    std::string fn_json = "/home/yuanjie/Projects/4sU_nanopore/chain.json";
    std::string fn_out = "/home/yuanjie/Projects/4sU_nanopore/chain_train.json";
    
    StepFitHMMGMM model;
    model.loadModel(fn_out);
    model.train(fn_data, 20, "", 50, 1500,  "te", 20);
    model.saveModel(fn_out);
    model.infer(fn_data, 10, "", fn_out_data);
}

int main() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_r(&now_c, &local_time);
    std::cout << "HMM: " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
    std::cout << "Active Threads: " << omp_get_num_threads() << std::endl;
    train();
    now = std::chrono::system_clock::now();
    now_c = std::chrono::system_clock::to_time_t(now);
    local_time;
    localtime_r(&now_c, &local_time);
    std::cout << "Finish: " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
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
