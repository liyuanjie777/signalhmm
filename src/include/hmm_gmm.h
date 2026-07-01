#include "fileio.h"
#include "hmm.h"
#include "model.h"
#include <vector>

class StepFitHMMGMM
{
public:
    StepFitHMMGMM(int kmer, int target, const char* method);
    ~StepFitHMMGMM() { clear(); };
    void clear();
    void preTrain(std::string& fn);
    void saveModel(std::string& fn) const;
    void train(const int batch, const int max_iter, const Real rate = 0.2);
    void infer(int batch);

private:
    std::string _filename;
    const char* _method;
    size_t _state_number;
    int _kmer;
    ReadsFile _readsfile;

    HMM* _hmm_model = nullptr;
    TransitionModel* _transition = nullptr;
    std::vector<EmissionModel*> _emission;
    std::vector<Real> _init_prob;
};