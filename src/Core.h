// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#ifndef CORE_H
#define CORE_H

#include <vector>
#include <queue>
#include <string>
#include "Neuron.h"
#include "Spike.h"
#include "Event_unit.h"
#include "Config.h"

#include "Adaptive.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Core {
public:
    Core(const std::string& param_file, const std::string& weights_file, const std::vector<int>& tau_values);
    Core(const Config& config);
    Core(const Config& config, const std::vector<int>& tau_values);
    Core(const Core& other); // 복사 생성자
    Core& operator=(const Core& other); // 복사 대입 연산자

    bool run();
    void load_spike_train(const std::vector<uint32_t>& spike_times, const std::vector<uint16_t>& neuron_indices);
    // Core.h에 추가
    void load_bias_spike_train(const std::vector<uint32_t>& spike_times, const std::vector<uint16_t>& neuron_indices);
    void save_recorded_spikes(const std::string& filename);
    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);

    void reset(); // 초기화 함수 추가

    bool enabling_train;
    uint32_t T_sim;
    uint8_t class_label;
    size_t train_index;

    size_t ET_N;
    size_t PTE_times;
    size_t PTE_slide;
    size_t PTE_range;
    size_t PTE_reg;
    double lr;

private:
    std::vector<std::vector<double>> W_in, W_res, W_out, W_bias, W_bias_out;
    std::vector<std::vector<bool>> W_fb;
    std::vector<Neuron> Neu_res, Neu_out, Neu_bias;
    std::vector<Neuron> Neu_out_1, Neu_out_2;
    
    std::vector<AdaptUnit> Neu_adapt;   // NEW
    double tau_adapt_default = 10000.0; // μs 단위 예시
    double b_step_default  = 0.1;       // V 단위 예시
    
    std::vector<size_t> Neu_acc;
    std::priority_queue<Spike> external_S_queue;
    std::priority_queue<Spike> internal_S_queue;
    std::priority_queue<Spike> S_vec_trace;
    std::priority_queue<Spike> S_vec_trace_delay;
    std::priority_queue<Event_unit> Event_queue;
    std::vector<Spike> S_vec_now;
    std::vector<Spike> S_vec_trace_now;
    std::vector<Spike> S_vec_trace_delay_now;
    std::vector<Event_unit> Event_vec_now;
    std::priority_queue<Event_unit> Event_queue_delay;
    uint32_t t_delay;
    size_t N_out_times;
    double alpha;
    double alpha_out;

    std::vector<uint32_t> recorded_times;
    std::vector<uint16_t> recorded_neuron_indices;

    bool run_loop();
    void record_spike(uint32_t time, int neuron_index);
};

#endif // CORE_H
