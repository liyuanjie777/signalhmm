#include "hmm_gmm.h"
#include "fileio.h"
#include "gaussianmixmodel.h"
#include "hmm.h"
#include "model.h"
#include "mymath.hpp"
#include "signalmath.hpp"
#include "sparsetransition.h"
#include "segment.h"
#include "cusum.hpp"
#include "json.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>

void StepFitHMMGMM::clear()
{
    if (_transition != nullptr) {
        delete _transition;
        _transition = nullptr;
    }
    for (auto& pair : _kmer_map) {
        delete pair.second;
        pair.second = nullptr;
    }
    _kmer_map.clear();
    _expand_seq.clear();
    _emission.clear();
    _init_prob.clear();
    _id2seq_id.clear();
    _id2label.clear();
    if (_hmm_model != nullptr)
    {
        delete _hmm_model;
        _hmm_model = nullptr;
    }
}

void StepFitHMMGMM::loadModel(const std::string& fn_json) {
    clear();
    _fn_src_json = fn_json;
    std::ifstream file(fn_json);
    if (!file.is_open()) {
        std::cerr << "can not read *.json!" << std::endl;
        return;
    }

    nlohmann::json data;
    file >> data;
    file.close();
    //read emission model
    for (auto& [key, val] : data["emission"].items()) {
        const int peak = val["peak"];
        _data_dim = val["dim"];
        std::vector<Real> params = val["params"];
        GaussianMixModel* em = new GaussianMixModel(peak, _data_dim);
        if (params.size() > 0) {
            em->setValues(params.data(), params.size());
        }
        _kmer_map[key] = em;
    }
    // get sequence length
    _state_number = data["nodes"].size();
    _id2seq_id.assign(_state_number, -1);
    _id2label.assign(_state_number, "*");
    int sequence_length = 0;
    for (auto& j_node : data["nodes"]) {
        const int seq_id = j_node["seq_id"];
        const int id = j_node["id"];
        const std::string label = j_node["label"];
        if (seq_id + 1 > sequence_length) {
            sequence_length = seq_id + 1;
        }
        _id2label[id] = label;
        _id2seq_id[id] = seq_id;
    }
    //set sequence map
    for (auto& j_node : data["nodes"]) {
        std::string label = j_node["label"];
        if (label != "*") {
            if (auto it = _expand_seq.find(label); it == _expand_seq.end()) {
                _expand_seq[label] = std::vector<std::vector<int>>(sequence_length);
            }
        }
    }
    if (_expand_seq.size() == 0) {
        _expand_seq[""] = std::vector<std::vector<int>>(sequence_length);
    }
    // build node
    _emission.resize(_state_number);
    for (auto& j_node : data["nodes"]) {
        const int id = j_node["id"];
        std::string kmer = j_node["kmer"];
        std::string label = j_node["label"];
        const int seq_id = j_node["seq_id"];
        if (label != "*") {
            _expand_seq[label][seq_id].push_back(id);
        }
        else {
            for (auto& it: _expand_seq) {
                it.second[seq_id].push_back(id);
            }
        }
        std::string lookup_key = kmer + "_" + label;
        if (auto it = _kmer_map.find(lookup_key); it != _kmer_map.end()) {
            _emission[id] = it->second;
        }
    }

    _init_prob.assign(_state_number, -std::numeric_limits<Real>::infinity());
    for (auto& [label, seq_matrix] : _expand_seq) {
        for (auto& node_ids : seq_matrix) {
            std::sort(node_ids.begin(), node_ids.end());
        }
        for (auto& id : seq_matrix[0]) {
            _init_prob[id] = 0.0;
        }
    }
    _transition = new SparseTransition();
    _hmm_model = new HMM(_emission, _transition, _init_prob);
    std::vector<Real> coo_val;
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    auto& edges = data["links"];
    size_t num_edges = edges.size();
    coo_x.reserve(num_edges);
    coo_y.reserve(num_edges);
    coo_val.reserve(num_edges);
    for (auto& edge : edges) {
        const int source = edge["source"];
        const int target = edge["target"];
        const Real prob = edge["prob"];
        coo_x.push_back(source);
        coo_y.push_back(target);
        coo_val.push_back(prob);
    }
    _transition->fill(coo_x.data(), coo_y.data(), coo_val.data(), coo_x.size(), _state_number);
    return;
}

void StepFitHMMGMM::saveModel(const std::string& fn_json_write) const {
    std::ifstream input_file(_fn_src_json);
    nlohmann::json data;
    input_file >> data;
    input_file.close();

    nlohmann::json emission_table = nlohmann::json::object();
    for (auto const& [key, em_ptr] : _kmer_map) {
        std::vector<Real> params = em_ptr->save();
        data["emission"][key]["params"] = params;
    }
    std::vector<int> coo_x(_transition->num_values());
    std::vector<int> coo_y(_transition->num_values());
    std::vector<Real> coo_val(_transition->num_values());
    _transition->save(coo_x.data(), coo_y.data(), coo_val.data());

    nlohmann::json links_array = nlohmann::json::array();
    for (size_t i = 0; i < coo_x.size(); ++i) {
        nlohmann::json j_edge;
        j_edge["source"] = coo_x[i];
        j_edge["target"] = coo_y[i];
        j_edge["prob"] = coo_val[i];
        links_array.push_back(j_edge);
    }
    data["links"] = links_array;

    std::ofstream output_file(fn_json_write);
    if (!output_file.is_open()) {
        std::cerr << "can not write *.json!" << std::endl;
    }
    output_file << std::setw(4) << data;
}

void StepFitHMMGMM::train(const std::string& fn_data, const int batch, const int max_iter, const int sampling, const char* method, const int max_band) {
    ReadsFile readsfile;
    readsfile.load(fn_data, false);
    Real resi = 0.0;
    Real prob_pre = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        int count = 0;
        std::vector<Read> tags = readsfile.readChunk(batch, 3000);
        while(tags.size() > 0) {
            std::vector<std::vector<Real>> datas;
            std::vector<std::vector<int>> mvs;
            std::vector<std::vector<std::vector<int>>> adj_lists;
            for (int i = 0; i < tags.size(); ++i) {
                if (tags[i].data.size() <= _state_number) {
                    continue;
                }
                //auto data_dim = sliding_mean_std(tags[i].data.data(), tags[i].data.size(), 13);
                datas.push_back(tags[i].data);
                std::vector<int> mv = tags[i].mv;
                mvs.push_back(mv);
                const std::string& label_name = tags[i].chrom;
                if (_expand_seq.size() == 1) {
                    adj_lists.push_back(compute_adj_list(_expand_seq[""], mv, max_band));
                }
                else {
                    adj_lists.push_back(compute_adj_list(_expand_seq[label_name], mv, max_band));
                }
                count++;
            }
            _hmm_model->EM_step(datas, _data_dim, method, adj_lists);
            tags = readsfile.readChunk(batch, 3000);
            if (sampling > 0 && count >= sampling) {break;}
        }
        readsfile.reset();
        const Real prob = _hmm_model->update(method);
        resi = prob - prob_pre;
        prob_pre = prob;
        printf("Iter: %d, Prob: %.4f, Residual: %.6f\n", iter, prob, resi);
    }
}
void StepFitHMMGMM::infer(const std::string& fn_data, const int batch, const std::string& fn_out) {
    std::ofstream tsv("tail_hmm.tsv", std::ios::out | std::ios::trunc);
    tsv << "read_id\tlabel\t";
    for (int i = 0; i < 50; ++i) {
        tsv << "c" << i;
        if (i < 49)
            tsv << "\t";
    }
    tsv << "\n";

    std::fstream ofile(fn_out, std::ios::binary | std::ios::trunc | std::ios::out);
    ofile.close();
    ReadsFile readsfile;
    readsfile.load(fn_data, false);
    std::vector<Read> tags = readsfile.readChunk(batch, 3000);
    while(tags.size() > 0) {
        std::vector<std::vector<Real>> datas;
        std::vector<std::vector<int>> mvs;
        std::vector<Real> distances;
        std::vector<std::string> write_str;
        for (auto tag : tags) {
            //auto data_dim = sliding_mean_std(tag.data.data(), tag.data.size(), 13);
            datas.push_back(tag.data);
            std::vector<int> mv;
            mv.reserve(tag.data.size());
            cumsum(tag.data, mv, 1.0,3.0);
            mvs.push_back(mv);
            write_str.push_back("");
        }
        _hmm_model->infer(datas, mvs, _data_dim, distances);
        for (int i = 0; i < mvs.size(); ++i) {
            for (int j = 0; j < mvs[i].size(); ++j) {
                mvs[i][j] = _id2seq_id[mvs[i][j]];
            }
            tags[i].mv = mvs[i];
            int label = tags[i].chrom.back() - '0';
            tsv << tags[i].uuid << "\t" << label << "\t";
            int segment_id = 0;
            int segment_start = 0;
            for (int j = 1; j < mvs[i].size(); ++j) {
                if (j == mvs[i].size() || mvs[i][j] != mvs[i][j - 1]) {
                    int end = j;
                    int n = std::min(50, end - segment_start);
                    Real sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                        sum += tags[i].data[segment_start + k];
                    }
                    Real mean = sum / n;
                    tsv << mean;
                    if (segment_id < 49)
                        tsv << "\t";
                    ++segment_id;
                    if (segment_id >= 50)
                        break;
                    segment_start = j;
                }
            }
            tsv << "\n";
        }
        readsfile.save(fn_out, tags);
        tags = readsfile.readChunk(batch, 3000);
        return;
    }
}

void StepFitHMMGMM::segment(const std::string& fn_data, const int batch, const std::string& fn_out) {
    std::fstream ofile(fn_out, std::ios::binary | std::ios::trunc | std::ios::out);
    ofile.close();
    ReadsFile readsfile;
    readsfile.load(fn_data, false);
    std::vector<Read> tags = readsfile.readChunk(batch, 1);
    while(tags.size() > 0) {
        std::vector<std::vector<Real>> datas;
        std::vector<std::vector<int>> labels(datas.size());
        for (int i = 0; i < tags.size(); ++i) {
            datas.push_back(tags[i].data);
            std::vector<int> mv;
            mv.reserve(datas[i].size());
            cumsum(datas[i], mv, 1.0,3.0);
            tags[i].mv = mv;
        }
        readsfile.save(fn_out, tags);
        return;
    }
}

void StepFitHMMGMM::score(const std::string& fn_data, const int batch, std::vector<std::string>& id, std::vector<Real>& label_real, std::vector<Real>& label_predict) {
    ReadsFile readsfile;
    readsfile.load(fn_data, false);
    std::vector<Read> tags = readsfile.readChunk(batch, 3000);
    while(tags.size() > 0) {
        std::vector<std::vector<Real>> datas;
        std::vector<std::vector<int>> mvs;
        std::vector<Real> distances;
        for (auto tag : tags) {
            datas.push_back(tag.data);
            std::vector<int> mv(tag.data.size());
            mvs.push_back(mv);
        }
        _hmm_model->infer(datas, mvs, _data_dim, distances);
        for (int i = 0; i < datas.size(); ++i) {
            id.push_back(tags[i].uuid);
            Real y = tags[i].chrom.back() - '0';
            label_real.push_back(y);
            y = _id2label[mvs[i][0]].back() - '0';
            label_predict.push_back(y);
        }
        tags = readsfile.readChunk(batch, 3000);
    }
}