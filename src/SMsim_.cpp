// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
#include <iomanip>
#include <getopt.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <bitset>

#include <arpa/inet.h>  // ntohl, ntohs 추가

#include "Core.h"

#ifndef __GIT_REV__
#define __GIT_REV__ "unknown"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// Load spike trains in parallel
void load_spike_trains_parallel(
    const std::vector<std::string>& file_paths,
    std::vector<std::vector<uint32_t>>& all_spike_times,
    std::vector<std::vector<uint16_t>>& all_neuron_indices,
    std::vector<uint8_t>& all_labels) 
{
    const int num_entries = file_paths.size();
    all_spike_times.clear();
    all_neuron_indices.clear();
    all_labels.clear();

    // 메모리 사전 할당
    all_spike_times.resize(num_entries);
    all_neuron_indices.resize(num_entries);
    all_labels.resize(num_entries);

    #pragma omp parallel for
    for(int i = 0; i < num_entries; ++i) {
        std::ifstream file(file_paths[i], std::ios::binary);
        if(!file.is_open()) {
            #pragma omp critical
            std::cerr << "파일 열기 실패: " << file_paths[i] << std::endl;
            continue;
        }

        // 헤더 파싱 -----------------------------------------------------------
        uint8_t header[13];
        file.read(reinterpret_cast<char*>(header), 13);

        // 헤더 구조: >BHIIBB (Big-Endian)
        uint8_t label = header[0];
        uint16_t data_index = ntohs(*reinterpret_cast<uint16_t*>(header + 1)); // 1-2바이트
        uint32_t uid = ntohl(*reinterpret_cast<uint32_t*>(header + 3));        // 3-6바이트
        uint32_t num_data_points = ntohl(*reinterpret_cast<uint32_t*>(header + 7)); // 7-10바이트
        uint8_t reserved1 = header[11];
        uint8_t reserved2 = header[12];


        // 레이블 디버깅 출력
        /*
        #pragma omp critical
        {
            std::cout << "파일: " << file_paths[i] 
                     << " | 레이블(숫자): " << static_cast<int>(label)
                     << " | 데이터포인트: " << num_data_points << std::endl;
        }
        */

        // 스파이크 데이터 읽기 -----------------------------------------------
        std::vector<uint32_t> local_spikes;
        std::vector<uint16_t> local_neurons;
        
        try {
            local_spikes.reserve(num_data_points);
            local_neurons.reserve(num_data_points);

            for(uint32_t j = 0; j < num_data_points; ++j) {
                uint32_t time_raw;
                uint16_t neuron_raw;
                
                // 시간 데이터 읽기 (4바이트 Big-Endian)
                file.read(reinterpret_cast<char*>(&time_raw), 4);
                time_raw = ntohl(time_raw); // Big-Endian → Little-Endian 변환
                double spike_time = static_cast<double>(time_raw); // μs → 초
                
                // 뉴런 인덱스 읽기 (2바이트 Big-Endian)
                file.read(reinterpret_cast<char*>(&neuron_raw), 2);
                neuron_raw = ntohs(neuron_raw);
                // std::cout<<spike_time<<' '<< neuron_raw << ' '<<'\n'; 이건 잘 나오는 거 확인함함
                
                local_spikes.emplace_back(time_raw);
                local_neurons.emplace_back(neuron_raw);
            }
            

            // 데이터 무결성 검증
            if(local_spikes.size() != num_data_points || 
            local_neurons.size() != num_data_points) {
                throw std::runtime_error("Data size mismatch");
            }
        } catch(...) {
            #pragma omp critical
            std::cerr << "Error reading: " << file_paths[i] << std::endl;
            continue;
        }

        #pragma omp critical
        {
            all_spike_times[i] = std::move(local_spikes);
            all_neuron_indices[i] = std::move(local_neurons);
            all_labels[i] = static_cast<int>(label);
            // std::cout<< all_spike_times[i][0] << '\n';
            // std::cout<< all_spike_times[i][10] << '\n';
            // std::cout<< all_neuron_indices[i][0] << '\n';
            // std::cout<< all_labels[i] << '\n';
            // std::cout<< static_cast<int>(label) << '\n';
            
            // 디버깅 출력
            /*
            std::cout << "File " << i << " | Spikes: " 
                    << all_spike_times[i].size() 
                    << " | Neurons: " 
                    << all_neuron_indices[i].size() 
                    << std::endl;
            */
        }
    }
}

// Format the duration into hours, minutes, and seconds
std::string format_duration(std::chrono::duration<double> duration) {
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(seconds);
    auto hours = std::chrono::duration_cast<std::chrono::hours>(minutes);
    seconds -= std::chrono::duration_cast<std::chrono::seconds>(minutes);
    minutes -= std::chrono::duration_cast<std::chrono::minutes>(hours);

    std::ostringstream oss;
    if (hours.count() > 0) {
        oss << hours.count() << "h ";
    }
    if (minutes.count() > 0 || hours.count() > 0) {
        oss << minutes.count() << "m ";
    }
    oss << seconds.count() << "s";
    return oss.str();
}

// Print the progress bar for current processing
void print_progress_bar(int current, int total, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
    int bar_width = 50;
    float progress = static_cast<float>(current) / total;
    int pos = bar_width * progress;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << format_duration(elapsed) << " elapsed)\r";
    std::cout.flush();
}

// Run the simulation and return the accuracy
// run_simulation 함수 수정
// 개선된 run_simulation 함수
double run_simulation(Core& core_template, 
    const std::vector<std::string>& all_files,
    int epoch,  // 사용되지 않는 매개변수 제거 가능
    const std::string& type, 
    int chunk_size = 2000) 
{
    const auto start_time = std::chrono::high_resolution_clock::now(); // start_time 선언 추가
    size_t total_processed = 0; // data_count 대체
    int correct_count = 0;
    bool enabling_train = (type == "train");

    for (size_t chunk_start = 0; chunk_start < all_files.size(); chunk_start += chunk_size) {
        auto chunk_end = std::min(chunk_start + chunk_size, all_files.size());
        std::vector<std::string> chunk_files(
            all_files.begin() + chunk_start,
            all_files.begin() + chunk_end
        );

        // 청크 데이터 처리
        std::vector<std::vector<uint32_t>> spike_times;
        std::vector<std::vector<uint16_t>> neuron_indices;
        std::vector<uint8_t> labels;
        load_spike_trains_parallel(chunk_files, spike_times, neuron_indices, labels);

        for (size_t i = 0; i < spike_times.size(); ++i) {
            core_template.reset();
            core_template.enabling_train = enabling_train;
            core_template.load_spike_train(spike_times[i], neuron_indices[i]);
            // std::cout<<"here?\n";
            core_template.class_label = static_cast<int>(labels[i]);

            // std::cout<<"here? label \n";
            // std::cout<<static_cast<int>(labels[i])<<'\n';

            bool is_correct = core_template.run();
            // std::cout<<"here???\n";
            if (is_correct) ++correct_count;

            // 진행률 업데이트
            ++total_processed;
            if (total_processed % 100 == 0) {
                print_progress_bar(total_processed, all_files.size(), start_time);
            }
        }
    }
    
    return static_cast<double>(correct_count) / total_processed; // data_count -> total_processed
}


// Print progress bar for epochs
void print_epoch_progress(int epoch, int total_epochs, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
    int bar_width = 50;
    float progress = static_cast<float>(epoch) / total_epochs;
    int pos = bar_width * progress;

    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% (" << format_duration(elapsed) << " elapsed)\r";
    std::cout.flush();
}

// Save accuracy to file
void save_accuracy_to_file(const std::string& filename, int epoch, double train_accuracy, double test_accuracy) {
    std::ofstream file;
    file.open(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open accuracy file");
    }

    file << epoch << "," << train_accuracy << "," << test_accuracy << "\n";
    file.close();
}

// main은 가능하면 건들이지 말아줘 나는 지금 데이터를 reservoir에 넣어주는 것만 관심이있어
int main(int argc, char *argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "Starting SMsim..." << std::endl;

    // File paths
    std::string param_file = "./init_parameters.json";
    std::string weights_file = "./init_weights.json";
    std::string tau_file = "./init_taus.json";

    // Load system parameters from JSON
    std::ifstream param_ifs(param_file);
    if (!param_ifs.is_open()) {
        throw std::runtime_error("Could not open parameter file");
    }
    json param_json;
    param_ifs >> param_json;

    int num_epochs = param_json["system_parameter"]["epoch"].get<int>();
    std::string base_train_file_path = param_json["system_parameter"]["training_file"].get<std::string>();
    std::string base_test_file_path = param_json["system_parameter"]["test_file"].get<std::string>();
    int T_sim = param_json["system_parameter"]["T_sim"].get<int>();
    double lr = param_json["system_parameter"]["lr"].get<double>();
    int N_chunks = param_json["system_parameter"]["N_chunks"].get<int>();
    int N_test_chunks = param_json["system_parameter"]["N_test_chunks"].get<int>();

    std::cout << "Loaded system parameters:" << std::endl;
    std::cout << "Epochs: " << num_epochs << std::endl;
    std::cout << "Training file path: " << base_train_file_path << std::endl;
    std::cout << "Test file path: " << base_test_file_path << std::endl;
    std::cout << "Simulation time (T_sim): " << T_sim << std::endl;

    std::cout << "version: " <<  __GIT_REV__ << std::endl;
    std::cout << "CXX: " <<  __VERSION__ << std::endl;

    // read tau file
    std::ifstream tau_ifs(tau_file);
    if (!tau_ifs.is_open()) {
        throw std::runtime_error("Could not open tau file");
    }
    json tau_json;
    tau_ifs >> tau_json;
    std::vector<int> tau_values = tau_json["tau"].get<std::vector<int>>();

    // generate accuracy file
    const std::string accuracy_file = "accuracy_log.csv";
    std::ofstream file(accuracy_file);
    if (file.is_open()) {
        file << "epoch, train_accuracy, test_accuracy \n";
        file.close();
    }

    // initialization of the core
    Core core_template(param_file, weights_file, tau_values);
    core_template.T_sim = T_sim;
    core_template.lr = lr;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        if (epoch > 0) {
            if (fs::exists("./training_weights.json")) {
                core_template.load_weights("./training_weights.json");
                std::cout << "Loaded weights for epoch " << epoch << std::endl;
            }
        }

        print_epoch_progress(epoch, num_epochs, program_start);

        std::cout << "\nStarting training epoch " << epoch << "...\n";

        int chunk_index = epoch % N_chunks;
        std::cout << chunk_index << "...\n";
#if defined(TRAIN_PHASE)
        // core_template.PTE_reg = (epoch / N_chunks) % core_template.PTE_times;
#endif
        std::vector<std::string> train_files, test_files;
        /*
        try {
            if(!fs::exists("./train_bin")) {
                throw std::runtime_error("train_bin디렉토리가 존재하지 않음");
            }
            
            for (const auto& entry : fs::directory_iterator("./train_bin")) {
                if (entry.path().extension() == ".bin") {
                    train_files.push_back(entry.path().string());
                }
            }
        } 
        catch (const fs::filesystem_error& e) {
            std::cerr << "\n[에러] 학습 데이터 접근 실패: " << e.what() << std::endl;
            std::cerr << "실행 경로 확인: " << fs::current_path() << std::endl;
            exit(EXIT_FAILURE);
        }
        for (const auto& entry : fs::directory_iterator("./test_bin")) {
            if (entry.path().extension() == ".bin") {
                test_files.push_back(entry.path().string());
            }
        }
        */
        
        try {
            if(!fs::exists("./train_bin_grouped")) {
                throw std::runtime_error("train_bin_grouped 디렉토리가 존재하지 않음");
            }
            
            for (const auto& entry : fs::directory_iterator("./train_bin_grouped")) {
                if (entry.path().extension() == ".bin") {
                    train_files.push_back(entry.path().string());
                }
            }
        } 
        catch (const fs::filesystem_error& e) {
            std::cerr << "\n[에러] 학습 데이터 접근 실패: " << e.what() << std::endl;
            std::cerr << "실행 경로 확인: " << fs::current_path() << std::endl;
            exit(EXIT_FAILURE);
        }
        for (const auto& entry : fs::directory_iterator("./test_bin_grouped")) {
            if (entry.path().extension() == ".bin") {
                test_files.push_back(entry.path().string());
            }
        }
        // std::cout<<"here \n";
        

        // 학습 단계
        double train_accuracy = run_simulation(core_template, train_files, epoch, "train");
        
        // 테스트 단계
        if (epoch % 5 == 0) {
            double test_accuracy = run_simulation(core_template, test_files, epoch, "test");
            save_accuracy_to_file(accuracy_file, epoch, train_accuracy*100, test_accuracy*100);
        }
        
        core_template.save_weights("./training_weights.json");

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "User time " << usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 << " sec" << std::endl;
    std::cout << "System time " << usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6 << " sec" << std::endl;
    std::cout << "maxurss " << usage.ru_maxrss / 1024 << " MB" << std::endl;

    auto program_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> program_duration = program_end - program_start;
    std::cout << "Total program duration: " << format_duration(program_duration) << ".\n";
}
}