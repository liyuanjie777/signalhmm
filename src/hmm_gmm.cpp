#include "hmm_gmm.h"
#include "fileio.h"
#include "gaussianmodel.h"
#include "hmm.h"
#include "model.h"
#include "mymath.hpp"
#include "sparsetransition.h"
#include <cstddef>
#include <fstream>
#include <string>


StepFitHMMGMM::StepFitHMMGMM(
    int state_number, int dim, const std::vector<Real>& koff, const char* method)
    : _state_number(state_number), _dim(dim), _method(method)
{
    for (int i = 0; i < _state_number; ++i)
    {
        _emission.push_back(new GaussianModel(dim));
    }
    _transition = new SparseTransition();
    std::vector<int> x;
    std::vector<int> y;
    std::vector<Real> val;
    for (int i = 0; i < _state_number; ++i)
    {
        for (int j = 0; j < _state_number; ++j)
        {
            if (i == j)
            {
                x.push_back(i);
                y.push_back(j);
                val.push_back(exp(-koff[i]));
            }
            else if (i + 1 == j)
            {
                x.push_back(i);
                y.push_back(j);
                val.push_back(1 - exp(-koff[i]));
            }
        }
    }
    int coo_num = x.size();
    _transition->setValues(x.data(), y.data(), val.data(), coo_num, _state_number);
    _init_prob.resize(_state_number, 1.0);
    _hmm_model = new HMM(_emission, _transition, _init_prob, _dim, _state_number, coo_num, _method);
}

StepFitHMMGMM::StepFitHMMGMM(std::string& fn)
{
    std::ifstream file(fn);
    std::string key;
    std::string method;
    Real value;
    int dim;
    int number;
    file >> method;
    file >> number;
    file >> dim;
    _dim = dim;
    _state_number = number;
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    std::vector<Real> coo_val;

    if (method == "gaussian")
    {
        int emit_para_num = dim * (dim + 1) + 2;
        for (int i = 0; i < number; ++i)
        {
            std::vector<Real> emit_val;
            for (int j = 0; j < emit_para_num; ++j)
            {
                file >> value;
                emit_val.push_back(value);
            }
            _emission.push_back(new GaussianModel(dim));
            _emission.back()->setValues(emit_val.data(), emit_val.size());
        }
        for (int i = 0; i < number; ++i)
        {
            file >> value;
            _init_prob.push_back(exp(value));
        }
        int coox, cooy;
        Real cooval;
        while (file >> coox >> cooy >> cooval)
        {
            coo_x.push_back(coox);
            coo_y.push_back(cooy);
            coo_val.push_back(cooval);
        }
        int coo_num = coo_x.size();
        _transition = new SparseTransition();
        _transition->setValues(coo_x.data(), coo_y.data(), coo_val.data(), coo_x.size(), number);
        _hmm_model =
            new HMM(_emission, _transition, _init_prob, _dim, _state_number, coo_num, _method);
    }
}

void StepFitHMMGMM::clear()
{
    if (_transition != nullptr)
    {
        delete _transition;
        _transition = nullptr;
    }
    for (int i = 0; i < _emission.size(); ++i)
    {
        delete _emission[i];
    }
    _emission.clear();
    _init_prob.clear();
    if (_hmm_model == nullptr)
    {
        delete _hmm_model;
        _hmm_model = nullptr;
    }
}

void StepFitHMMGMM::loadData(std::string& fn)
{
    _filename = fn;
    _readsfile.load(fn, 500, true);
}


void StepFitHMMGMM::saveModel(std::string& fn)
{
    std::string method = "gaussian";
    std::ofstream file(fn);
    if (method == "gaussian")
    {
        file << method << " " << _state_number << " " << _dim << "\n";
        for (int i = 0; i < _state_number; ++i)
        {
            std::vector<Real> val = _emission[i]->save();
            for (int j = 0; j < val.size(); ++j)
            {
                file << val[j] << " ";
            }
            file << "\n";
        }
        for (int i = 0; i < _state_number; ++i)
        {
            file << _init_prob[i] << " ";
        }
        file << "\n";
        std::vector<int> coo_x(_transition->num_values());
        std::vector<int> coo_y(_transition->num_values());
        std::vector<Real> coo_val(_transition->num_values());
        _transition->save(coo_x.data(), coo_y.data(), coo_val.data());
        for (int i = 0; i < coo_x.size(); ++i)
        {
            file << coo_x[i] << " " << coo_y[i] << " " << coo_val[i] << std::endl;
        }
    }
}

void StepFitHMMGMM::train(int batch, int max_iter, Real rate)
{
    int reads_number = _readsfile.readsNumber;
    std::vector<std::vector<int>> batch_id;
    std::vector<Real> x_data;
    std::vector<size_t> x_batch;
    for (int i = 0; i < 10; ++i)
    {
        if (i % batch == 0)
        {
            batch_id.push_back(std::vector<int>());
        }
        batch_id.back().push_back(i);
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        for (int i = 0; i < batch_id.size(); ++i)
        {
            //_readsfile.getReads(batch_id[i], x_data, _dim, x_batch);
            _hmm_model->train_step(x_data.data(), x_batch.data(), x_batch.size() - 1);
        }
        Real resi = _hmm_model->update(rate);
        printf("iter: %d, prob: %f\n", iter, resi);
    }
}
void StepFitHMMGMM::infer(int batch)
{
    std::string fn = "/home/yuanjie/code/MapSignal/src/test_result.dat";
    int reads_number = _readsfile.readsNumber;
    std::vector<std::vector<int>> batch_id;
    std::vector<Real> x_data;
    std::vector<int> x_label;
    std::vector<size_t> x_batch;
    for (int i = 0; i < reads_number; ++i)
    {
        if (i % batch == 0)
        {
            batch_id.push_back(std::vector<int>());
        }
        batch_id.back().push_back(i);
    }

    for (int i = 0; i < batch_id.size(); ++i)
    {
        //_readsfile.getReads(batch_id[i], x_data, _dim, x_batch);
        int num_batch = x_batch.size() - 1;
        x_label.resize(x_batch[num_batch], 0);
        _hmm_model->infer(x_data.data(), x_label.data(), x_batch.data(), num_batch);
        std::vector<Real> odata(_emission.size(), 0);
        for (int j = 0; j < num_batch; ++j)
        {
            odata.resize(_emission.size(), 0);
            int start_pos = x_batch[j];
            int end_pos = start_pos + 1;
            std::vector<Real> mean(_dim);
            for (size_t t = x_batch[j]; t < x_batch[j + 1]; ++t)
            {
                if (x_label[t] == x_label[start_pos])
                {
                    end_pos = t + 1;
                }
                else
                {
                    vec_mean(
                        x_data.data() + start_pos * _dim, end_pos - start_pos, mean.data(), _dim);
                    _emission[x_label[start_pos]]->score(
                        mean.data(), &odata[x_label[start_pos]], 1);
                    start_pos = t;
                    end_pos = t + 1;
                }
            }
            //_readsfile.saveReads(fn, odata, batch_id[i][j]);
        }
        printf("infer: %d%\n", i * batch * 100 / reads_number);
    }
}