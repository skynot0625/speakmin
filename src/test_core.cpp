// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include "Core.h"
#include "Neuron.h"
#include "Spike.h"
#include "Event_unit.h"

// Minimal config struct for testing
Config make_test_config() {
    Config config;
    config.N_in = 1;
    config.N_res = 2;
    config.N_out = 1;
    config.N_bias = 0;
    config.N_class = 1;
    config.N_out_times = 1;
    config.PTE_slide = 1;
    config.PTE_times = 1;
    config.PTE_range = 1;
    config.ET_N = 1;
    config.T_sim = 10;
    config.t_delay = 1;
    config.V_init = 0.0;
    config.tau_out = 1000.0;
    config.tau_out_1 = 1000.0;
    config.tau_out_2 = 1000.0;
    config.tau_out_3 = 1000.0;
    config.tau_out_4 = 1000.0;
    config.tau_out_5 = 1000.0;
    config.V_th = 1.0;
    config.V_bot = -2.0;
    config.V_reset = 0.0;
    config.alpha = 1.0;
    config.alpha_out = 1.0;
    config.SG_window = 0.5;

    // 1 input, 2 reservoir, 1 output
    config.W_in = {{1.0, 0.0}}; // input to res0, not res1
    config.W_res = {{0.0, 1.0}, {1.0, 0.0}}; // res0->res1, res1->res0
    config.W_out = {{1.0}, {1.0}}; // both res to out
    config.W_fb = {{false}}; // not used
    config.W_bias = {};
    config.W_bias_out = {};
    return config;
}

int main() {
    std::cout << "=== Spiking Reservoir Unit Test ===" << std::endl;

    // Use fixed tau for both neurons
    std::vector<int> tau_values = {1000, 1000};

    Config config = make_test_config();
    Core core(config, tau_values);

    // Inject a spike at t=0 to input neuron 0
    std::vector<uint32_t> spike_times = {0};
    std::vector<uint16_t> neuron_indices = {0};
    core.load_spike_train(spike_times, neuron_indices);

    // Run for a few steps, manually step through the event queue
    core.T_sim = 5;
    core.enabling_train = false;
    core.class_label = 0;

    // Run the simulation (should propagate spikes through the reservoir)
    bool result = core.run();

    // Print out the result
    std::cout << "Simulation result (is_correct): " << result << std::endl;

    // Optionally, print the state of each neuron
    // (Assumes public access or add getters if needed)
    // For demonstration, print after simulation
    std::cout << "Test complete. Check console output for spike propagation and neuron state." << std::endl;

    return 0;
}
