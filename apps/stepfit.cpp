//
// Created by yuanjie on 10/6/26.
//

#include <fstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include "hmm_gmm.h"
#include "stepfit.h"

int train(int argc, char* argv[]) {
    int batch_size = 1;
    int sampling_number = 1;
    int max_iter = 10;
    int nband = 50;
    std::string method = "et";
    std::string input;
    std::string model;
    std::string output;
    int i = 1;
    while (i < argc) {
        const std::string arg = argv[i];
        if (arg != "--input" && arg != "--model" && arg != "--output" && arg != "--batch-size" && arg != "--sampling-number" && arg != "--iter-max" && arg != "--band" && arg != "--method") {
            throw std::invalid_argument("Unknown option: " + arg);
        }
        const std::string value = argv[++i];
        ++i;
        if (value.empty() || value.rfind("--", 0) == 0) {
            throw std::invalid_argument("Missing value for " + arg);
        }
        if (arg == "--input")
            input = value;
        else if (arg == "--model")
            model = value;
        else if (arg == "--output")
            output = value;
        else if (arg == "--batch-size")
            batch_size = std::stoull(value);
        else if (arg == "--sampling-number")
            sampling_number = std::stoull(value);
        else if (arg == "--iter-max")
            max_iter = std::stoull(value);
        else if (arg == "--band")
            nband = std::stoull(value);
        else if (arg == "--method")
            method = value;
    }
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_time;
    localtime_r(&now_c, &local_time);
    std::cout << "HMM: " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
    std::cout << "Active Threads: " << omp_get_num_threads() << std::endl;
    StepFitHMMGMM hmmmodel;
    hmmmodel.loadModel(model);
    hmmmodel.train(input, batch_size, max_iter, sampling_number,  method.c_str(), nband);
    hmmmodel.saveModel(output);
    now = std::chrono::system_clock::now();
    now_c = std::chrono::system_clock::to_time_t(now);
    local_time;
    localtime_r(&now_c, &local_time);
    std::cout << "Finish: " << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S") << std::endl;
    return 1;
}