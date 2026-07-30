#include "fileio.h"
#include "hmm.h"
#include "model.h"
#include <vector>

class StepFitHMMGMM
{
public:
    StepFitHMMGMM(const char* method) : _method(method) {};
    ~StepFitHMMGMM() { clear(); };
    void clear();
    void allocateModel(const std::string& sequence, const int extend_number, const int kmer);
    void loadModel(const std::string& fn_model, const std::string& sequence, const int extend_number, const int kmer);
    void saveModel(std::string& fn) const;
    void train(const int batch, const int max_iter, const Real rate = 0.2);
    void infer(int batch);

private:
    const char* _method;
    std::vector<std::string> _kmer_name;

    std::string _filename;
    int _state_number;
    int _extend_number;
    std::string _ref_sequence;
    ReadsFile _readsfile;

    HMM* _hmm_model = nullptr;
    TransitionModel* _transition = nullptr;
    std::vector<EmissionModel*> _emission;
    std::vector<Real> _init_prob;
};