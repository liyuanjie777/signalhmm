/*
 *  Project: Signal Map
 *  Author:  Yuanjie Li
 *  Date:    2025-08-07
 *  License: MIT
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "fileio.h"
#include "mymath.hpp"
#include "signalmath.hpp"
#include "hmm_gmm.h"


void test() {
    std::string fn = "/home/yuanjie/Projects/4sU_nanopore/hiPSC-CM-cDNA-IVT_UTP.dat";
    ReadsFile reads = ReadsFile();
    reads.load(fn, 100, true);
    std::string seq;
    std::vector<Real> data;
    std::vector<char> mv;
    
    for(int i=0; i < 1; i++){
        reads.read(seq, data, mv);
        std::vector<Real> data2;
        std::vector<size_t> batch;
        std::vector<ChunkInfo> chunk;
        reads.readChunk(data2, batch, chunk);
        MAD(data.data(), data.size());
        auto a = kmer_to_index(seq.c_str(), seq.size(), 7);
        
        std::cout<<seq;
        std::cout<<seq.size();
        continue;
    }
}

int main()
{
    int nstate = 20;
    std::vector<Real> koff(nstate, 0.01);
    std::string fn = "/home/yuanjie/Projects/4sU_nanopore/hiPSC-CM-c DNA-IVT_UTP.dat";
    //std::string fn_save = "/home/yuanjie/Projects/MapSignal/src/test.model";
    // StepFitHMMGMM model(nstate, 1, koff, "etp");
    // model.loadData(fn);
    // model.train(20, 10);
    // model.saveModel(fn_save);
    test();
    //StepFitHMMGMM model_test(fn_save);
    //model_test.loadData(fn);
    //model_test.infer(10);
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
