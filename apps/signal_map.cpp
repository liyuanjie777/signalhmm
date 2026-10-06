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
#include "view.h"
#include "stepfit.h"
/*
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
}/

 */


int main(int argc, char* argv[]) {
    const std::string command = argv[1];
    if (command == "viewbin") {
        return viewbin(argc - 1, argv + 1);
    }

    if (command == "train") {
        return train(argc - 1, argv + 1);
    }

    if (command == "--help" || command == "-h") {
        std::cout <<  R"(
        Usage:
          mapsignal viewbin <file> summary <all|N>
          mapsignal viewbin <file> show <index>

        Commands:
          summary all    Print the summary for all
          summary N      Print the summary for N
          show index     Print the current and move table for index item

        Examples:
          mapsignal viewbin ./reads.bin summary all
          mapsignal viewbin ./reads.bin summary 10
          mapsignal viewbin ./reads.bin show 5
        )";

        std::cout << R"(
        Usage:
          mapsignal train --input <file> --model <file> --output <file> [options]

        Required:
          --input FILE            input data file .bin
          --model FILE            model file .json
          --output FILE           output model file .json

        Options:
          --batch-size N          size of batch, default 1
          --sampling-number N     reads number for training，default 100
          --iter-max N            max iter number，defualt 10
          --band N            band width for gamma matrix, default 20
          --method STRING         update method, defualt te
                                    e：emission update
                                    t：transition update
                                    p：initial probability update
                                  e.g. te、tep
        Example:
          mapsignal tain --data ./reads.bin --model ./model.json --out ./trained.json \
              --batch-size 20 --sampling-number 1500 \
              --iter-max 50 --band 20 --method te
        )";
        return 0;
    }
    std::cerr << "Unknown command: " << command << '\n';
    return 1;
}

