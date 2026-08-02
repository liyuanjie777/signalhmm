#include "fileio.h"
#include "hmm.h"
#include "model.h"
#include <vector>

class StepFitHMMGMM
{
public:
    ~StepFitHMMGMM() { clear(); };
    void clear();
    void loadData(const std::string& fn, const bool shuffle) { _readsfile.load(fn, shuffle); };
    void allocateModel(const std::string& fn_fa, const std::string& chrom, const int extend_number, const int peak);
    void loadModel(const std::string& fn_model, const std::string& fn_fa, const std::string& chrom, const int extend_number, const int peak);
    void saveModel(std::string& fn) const;
    void train(const int batch, const int data_dim, const int max_iter, const int sampling, const Real rate, const char* method, const int max_band);
    void infer(const int batch, const int data_dim, const std::string& fn_out);

private:
    int _state_number;
    int _extend_number;
    int _peak;
    ReadsFile _readsfile;
    std::string _sequence;
    std::string _chrom;

    HMM* _hmm_model = nullptr;
    TransitionModel* _transition = nullptr;
    std::vector<EmissionModel*> _emission;
    std::vector<Real> _init_prob;
};