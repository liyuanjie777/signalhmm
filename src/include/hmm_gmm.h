#include "fileio.h"
#include "hmm.h"
#include "model.h"
#include <vector>

class StepFitHMMGMM
{
public:
    ~StepFitHMMGMM() { clear(); };
    void clear();
    void loadModel(const std::string& fn_json);
    void saveModel(const std::string& fn_json_write) const;
    void train(const std::string& fn_data, const int batch, const int max_iter, const int sampling, const char* method, const int max_band);
    void infer(const std::string& fn_data, const int batch, const std::string& fn_out);
    void segment(const std::string& fn_data, const int batch, const std::string& fn_out);
    void score(const std::string& fn_data, const int batch, std::vector<std::string>& id, std::vector<Real>& label_real, std::vector<Real>& label_predict);

private:
    int _state_number;
    int _data_dim;
    std::unordered_map<std::string, std::vector<std::vector<int>>> _expand_seq;

    HMM* _hmm_model = nullptr;
    TransitionModel* _transition = nullptr;
    std::vector<EmissionModel*> _emission;
    std::vector<int> _id2seq_id;
    std::vector<std::string> _id2label;
    std::unordered_map<std::string, EmissionModel*> _kmer_map;
    std::vector<Real> _init_prob;
    std::string _fn_src_json;
};