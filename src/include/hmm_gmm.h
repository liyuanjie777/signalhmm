#include "fileio.h"
#include "hmm.h"
#include "model.h"
#include <vector>

class StepFitHMMGMM
{
public:
    StepFitHMMGMM(int state_number, int dim, const std::vector<Real>& koff, const char* method);
    StepFitHMMGMM(const std::vector<EmissionModel*>& emissions, int kmer, Real koff, const char* method);
    ~StepFitHMMGMM() { clear(); };
    void clear();
    void loadData(std::string& fn);
    void saveModel(std::string& fn);
    void train(int batch, int max_iter, Real rate = 0.2);
    void infer(int batch);

private:
    std::string _filename;
    const char* _method;
    int _state_number;
    int _dim;
    ReadsFile _readsfile;

    HMM* _hmm_model = nullptr;
    TransitionModel* _transition = nullptr;
    std::vector<EmissionModel*> _emission;
    std::vector<Real> _init_prob;
};