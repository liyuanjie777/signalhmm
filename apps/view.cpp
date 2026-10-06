//
// Created by yuanjie on 10/6/26.
//
#include  <iostream>
#include <iomanip>
#include "fileio.h"
#include "view.h"

int viewbin(int argc, char* argv[]) {
    const std::string mode = argv[2];
    if (mode != "summary" && mode != "show")
        throw std::invalid_argument("Expected summary or show");
    const bool summary = mode == "summary";
    const bool all = summary && std::string(argv[3]) == "all";
    int requested = all ? 0 : std::stoull(argv[3]);
    ReadsFile files;
    files.load(argv[1], false);
    if (summary) {
        std::cout << "INDEX\tUUID\tCHROM\tSIGNAL_COUNT\n";
        if (requested == 0) requested = files.size();
        for (int i = 0; i < requested; ++i) {
            Read read = files.read(i);
            std::cout << read.index << '\t' << read.uuid << '\t' << read.chrom << '\t' << read.data.size() << '\n';
        }
    }
    else {
        Read read = files.read(requested);
        std::cout << "Index: " << read.index << "\nUUID: " << read.uuid << "\nChrom: " << read.chrom << "\n";
        const auto flags = std::cout.flags();
        const auto precision = std::cout.precision();
        std::cout << std::right << std::setw(14) << "Signal" << std::setw(12) << "MV" << '\n' << std::fixed << std::setprecision(3);
        const size_t n = read.data.size();
        for (size_t i = 0; i < n; ++i) {
            if (i < read.data.size())
                std::cout << std::setw(14) << read.data[i];
            else
                std::cout << std::setw(14) << "-";

            if (i < read.mv.size())
                std::cout << std::setw(12) << read.mv[i];
            else
                std::cout << std::setw(12) << "-";
            std::cout << '\n';
        }
        std::cout.flags(flags);
        std::cout.precision(precision);
    }
    return 1;
}