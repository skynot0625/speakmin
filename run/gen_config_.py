# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import json
import os

# Set the random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Define the dimensions for the weights and parameters
'''
num_neu_in = 16
num_neu_res = 470
num_class = 10
'''
'''
num_neu_in = 32
num_neu_res = 470
num_class = 10
'''
'''
num_neu_in = 32
num_neu_res = 470
num_class = 10
'''

num_neu_in = 64
num_neu_res = 512 - 80
num_class = 20
# num_class = 6
# num_class = 10

num_out_times = 4
# num_out_times = 1
# 2, 4, 10
num_neu_out = num_class * num_out_times # 10
# 20, 40, 100

num_neu_bias = 40
# num_neu_bias = 60

# Function to quantize weights to 8-bit based on custom range
def quantize_weights(weights, min_val, max_val):
    quantized = np.round((weights - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    dequantized = quantized / 255 * (max_val - min_val) + min_val
    return dequantized

# Generate random weights for W_res
W_res = rng.uniform(-1, 1, (num_neu_res, num_neu_res))

W_res = np.zeros((num_neu_res, num_neu_res))
num_neg_rows = num_neu_res // 5
num_pos_rows = num_neu_res - num_neg_rows
# 음수 행 (-1 ~ 0)
W_res[:num_neg_rows] = rng.uniform(-1, 0, (num_neg_rows, num_neu_res))
# 양수 행 (0 ~ 1)
W_res[num_neg_rows:] = rng.uniform(0, 1, (num_pos_rows, num_neu_res))

W_res = rng.uniform(-1, 1, (num_neu_res, num_neu_res))

sparsity = -1
mask = rng.random(size=(num_neu_res, num_neu_res)) > sparsity
W_res = W_res * mask  # 원소별 곱셈으로 마스크 적용
np.fill_diagonal(W_res, 0)
rho_W_res = max(abs(np.linalg.eigvals(W_res)))
print(rho_W_res) 

# Calculate in_scale for W_in
in_scale = 1 / rho_W_res * 1.25
print(in_scale)
W_in = rng.uniform(-1, 1, (num_neu_in, num_neu_res))

mask = rng.random(size=(num_neu_in, num_neu_res)) > sparsity
W_in = W_in * mask  # 원소별 곱셈으로 마스크 적용

def xavier_init_uniform(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return rng.uniform(-limit, limit, size=(input_size, output_size))

def xavier_init_normal(input_size, output_size):
    stddev = np.sqrt(2 / (input_size + output_size))
    return rng.normal(0, stddev, size=(input_size, output_size))

W_out = rng.uniform(-1, 1, (num_neu_res, num_neu_out))
W_bias = rng.uniform(-1, 1, (num_neu_bias, num_neu_res))

W_bias_out = rng.uniform(-1, 1, (num_neu_bias, num_neu_out))

W_fb = rng.uniform(-1, 1, (num_neu_res, num_neu_out)) > 0

W_in_quantized = W_in.tolist()
W_res_quantized = W_res.tolist()
W_out_quantized = W_out.tolist()
W_bias_quantized = W_bias.tolist()
W_bias_out_quantized = W_bias_out.tolist()
W_fb = W_fb.astype(bool).tolist()

# Log-normal distribution parameters
mu, sigma = 0.5, 0.5
mu, sigma = 0.5, 1
num_samples = num_neu_res

# Generate initial log-normal samples
tau_samples = rng.lognormal(mu, sigma, num_samples)

# Define the boundaries for resampling
lower_bound = 0.0
upper_bound = 6.0

# Separate samples into bins
inside_bounds = (tau_samples >= lower_bound) & (tau_samples <= upper_bound)
tau_samples_resampled = tau_samples[inside_bounds]

while np.sum(~inside_bounds) > 0:
    resampled_values = rng.lognormal(mu, sigma, np.sum(~inside_bounds))
    inside_bounds_new = (resampled_values >= lower_bound) & (resampled_values <= upper_bound)
    tau_samples_resampled = np.concatenate((tau_samples_resampled, resampled_values[inside_bounds_new]))
    inside_bounds = inside_bounds_new

# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 2500
tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 500 # alpha=5
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]) * 2000 # alpha=5
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]) * 1000 # alpha=5
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]) * 500 # alpha=5
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15]) * 1000 # alpha=5
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 2000 # alpha=1
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8]) * 500 # alpha=1
bins = np.linspace(0.0, 6.0, len(tau_values) + 1)
tau_bins = np.digitize(tau_samples_resampled, bins)
tau_bins = np.clip(tau_bins - 1, 0, len(tau_values) - 1)
tau_array_hetero = tau_values[tau_bins]

# 로그정규분포 파라미터 설정
median = 4.0  # 목표 중간값 (ms)
mu = np.log(median)  # median = exp(mu)
sigma = 1          # 분산(폭) 조절

# 타겟 tau 값들 설정
tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 1000

# 샘플링 및 quantization 함수
def generate_quantized_tau_samples(num_samples=512, lower_bound=0.0, upper_bound=6.0):
    """로그정규분포 샘플링 후 tau_values로 quantization"""
    
    # 초기 샘플 생성
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    
    # 경계 내의 샘플만 선택하고 필요시 재샘플링
    inside_bounds = (samples >= lower_bound) & (samples <= upper_bound)
    tau_samples_resampled = samples[inside_bounds]
    
    while len(tau_samples_resampled) < num_samples:
        additional_samples = np.random.lognormal(mean=mu, sigma=sigma, 
                                               size=num_samples - len(tau_samples_resampled))
        additional_bounded = additional_samples[(additional_samples >= lower_bound) & 
                                              (additional_samples <= upper_bound)]
        tau_samples_resampled = np.concatenate([tau_samples_resampled, additional_bounded])
    
    # 정확히 필요한 개수만큼 자르기
    tau_samples_resampled = tau_samples_resampled[:num_samples]
    
    # bins 생성 및 quantization
    bins = np.linspace(lower_bound, upper_bound, len(tau_values) + 1)
    tau_bins = np.digitize(tau_samples_resampled, bins)
    tau_bins = np.clip(tau_bins - 1, 0, len(tau_values) - 1)
    
    # tau_values에 매핑하여 최종 결과 생성
    tau_array_hetero = tau_values[tau_bins]
    
    return tau_array_hetero

# 사용 예시
num_samples = num_neu_res
tau_array_hetero = generate_quantized_tau_samples(num_samples)

# 결과 확인
print(f"Generated {len(tau_array_hetero)} samples")
print(f"Unique tau values: {np.unique(tau_array_hetero)}")
unique_vals, counts = np.unique(tau_array_hetero, return_counts=True)
print(f"Distribution: {dict(zip(unique_vals, counts))}")

# uniform
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8]) * 2000
# tau_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56]) * 500
# tau_values = np.array([1, 4, 8, 24, 56]) * 500
# tau_array_hetero = rng.choice(tau_values, size=num_neu_res)

# Results validation
mean_actual = np.mean(tau_array_hetero)
print(f"Actual mean: {mean_actual}")

unique, counts = np.unique(tau_array_hetero, return_counts=True)
for value, count in zip(unique, counts):
    print(f"Value {value}: {count} samples")

tau_homo_value = 4000.0
tau_array_homo = np.full(num_neu_res, tau_homo_value)
print(tau_homo_value)

tau_type = "hetero" # "hetero" or "homo"

if tau_type == "hetero":
    tau_array = tau_array_hetero
else:
    tau_array = tau_array_homo

# Define the core parameters dictionary
core_parameters = {
    "t_delay": 1,
    "V_init": 0.0,
    "tau_out": 100000.0,
    "tau_out_1": 4 * 4000.0,
    "tau_out_2": 7 * 4000.0,
    "tau_out_3": 32 * 4000.0,
    "tau_out_4": 56* 4000.0,
    # "tau_out_1": 1 * 500.0,
    # "tau_out_2": 8* 500.0,
    # "tau_out_3": 4 * 500.0,
    # "tau_out_4": 32* 500.0,
    # "tau_out_1": 1 * 4000.0,
    # "tau_out_2": 2 * 4000.0,
    # "tau_out_3": 4 * 4000.0,
    # "tau_out_4": 8 * 4000.0,
    "tau_out_1": 4 * 2000.0,
    "tau_out_2": 7 * 2000.0,
    "tau_out_3": 32 * 2000.0,
    "tau_out_4": 56* 2000.0,
    # "tau_out_1": 2 * 1000.0,
    # "tau_out_2": 2 * 2000.0,
    # "tau_out_3": 56 * 1000.0,
    # "tau_out_4": 56* 2000.0,
    ## "tau_out_1": 16 * 1000.0,
    ## "tau_out_2": 2 * 1000.0,
    ## "tau_out_3": 4 * 1000.0,
    ## "tau_out_4": 8* 1000.0,
    "tau_out_1": 2000.0,
    "tau_out_2": 16000.0,
    "tau_out_3": 4000.0,
    "tau_out_4": 10000.0,
    "V_bot": -2.0,
    "V_th": 1.0,
    "V_reset": 0.0,
    "t_ref": 1000,
    "alpha": 0.1,
    "alpha_out": 0.1,
    "SG_window": 0.5,
    "N_in": num_neu_in,
    "N_res": num_neu_res,
    "N_out": num_neu_out,
    "N_bias": num_neu_bias,
    "N_class": num_class,
    "N_out_times": num_out_times,
    "PTE_slide": 1,
    "PTE_times": 1,
    "PTE_range": 1,
    "ET_N": 100,
}

# Define the system parameters dictionary
system_parameters = {
    "T_sim": 1000000000,
    "epoch": 1000,                                               # Example epoch value
    "lr": 0.004,                                                # same as conductance steps. This is for 8bits ~ 1/250.
    # "test_file": "../tools/speech-to-spikes/gen_spike/test.bin",    # Replace with the actual test file path
    "test_file": "../tools/speech-to-spikes/gen_spike/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0124_all_32_5_10ms_pre_norm/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0202_all_32_2_no_pre_norm_mixed/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0119_all_32_2_no_pre_norm_pre_norm_0.68/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0117_all_32_5_10ms_pre_norm_pre/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0130_all_32_1_no_pre_0.68_no_align/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0202_all_32_2_16ms_pre_norm_mixed/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0206_6select_2_no_pre_norm_pre_norm_mixed/test",
    # "test_file": "../tools/speech-to-spikes/gen_spike/0128_all_32_1_no_pre_norm/test",

    # "test_file": "../tools/speech-to-spikes/gen_spike/all_32_2/test.bin",    # Replace with the actual test file path
    # "test_file": "../tools/speech-to-spikes/gen_spike/all_32_5/test.bin",    # Replace with the actual test file path
    # "test_file": "../tools/speech-to-spikes/gen_spike/all_32_NONE_10ms/test.bin",
    "training_file": "../tools/speech-to-spikes/gen_spike/train",   # Replace with the actual training file path
    # "training_file": "../tools/speech-to-spikes/gen_spike/0124_all_32_5_10ms_pre_norm/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0202_all_32_2_no_pre_norm_mixed/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0115_all_32_2_no_pre_norm/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0119_all_32_2_no_pre_norm_pre_norm_0.68/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0117_all_32_5_10ms_pre_norm_pre/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0130_all_32_1_no_pre_0.68_no_align/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0202_all_32_2_16ms_pre_norm_mixed/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0206_6select_2_no_pre_norm_pre_norm_mixed/train",
    # "training_file": "../tools/speech-to-spikes/gen_spike/0206_10num_2_no_pre_norm_pre_norm_mixed/train",
    #"training_file": "../tools/speech-to-spikes/gen_spike/all_32_2/train",   # Replace with the actual training file path
    # "training_file": "../tools/speech-to-spikes/gen_spike/all_32_5/train",   # Replace with the actual training file path
    # "training_file": "../tools/speech-to-spikes/gen_spike/0128_all_32_1_no_pre_norm/train",
    "N_chunks": 10,                             # you can devide training dataset as 'chunk'
    "N_test_chunks": 1
}

# Combine system and core parameters into a single dictionary
parameters = {
    "system_parameter": system_parameters,
    "core_parameter": core_parameters
}

# Define the weights dictionary with quantized weights
weights = {
    "W_in": W_in_quantized,
    "W_res": W_res_quantized,
    "W_out": W_out_quantized,
    "W_bias": W_bias_quantized,
    "W_bias_out": W_bias_out_quantized,
    "W_fb": W_fb
}

# Paths to the JSON files
parameters_path = './init_parameters.json'
weights_path = './init_weights.json'
tau_path = './init_taus.json'

# Save the parameters to the init_parameters.json file
with open(parameters_path, 'w') as f:
    json.dump(parameters, f, indent=4)
print(f"Parameters file saved to {parameters_path}")

# Save the weights to the init_weights.json file
with open(weights_path, 'w') as f:
    json.dump(weights, f, indent=4)
print(f"Weights file saved to {weights_path}")

# Save the tau values to the init_taus.json file
tau_dict = {"tau": tau_array.tolist()}
with open(tau_path, 'w') as f:
    json.dump(tau_dict, f, indent=4)
print(f"Tau file saved to {tau_path}")

# Check the distribution of tau values
tau_distribution = {tau: list(tau_array).count(tau) for tau in tau_values}
print(f"Tau distribution: {tau_distribution}")

# Print only the variables from the parameters file
print("\nParameters Variables:")
for key, value in parameters.items():
    print(f"{key}: {value}")
