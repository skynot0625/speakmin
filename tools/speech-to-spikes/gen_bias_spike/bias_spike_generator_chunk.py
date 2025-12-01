import numpy as np
import struct
import random
import os

def generate_bias_chunk_file(output_file, hz, duration_us, num_bias_neurons, 
                             num_samples_in_file, start_seed_offset):
    """
    하나의 바이너리 파일 안에 num_samples_in_file 만큼의 스파이크 데이터를 연속으로 저장합니다.
    C++의 load_spike_trains_parallel 함수가 읽을 수 있는 구조입니다.
    """
    
    duration_s = duration_us / 1e6
    
    # 바이너리 파일 열기 (쓰기 모드)
    with open(output_file, 'wb') as f:
        
        for i in range(num_samples_in_file):
            # 1. 랜덤 시드 설정 (샘플마다 다르게 설정하여 패턴 다양화)
            current_seed = start_seed_offset + i
            np.random.seed(current_seed)
            random.seed(current_seed)
            
            # 2. 스파이크 생성 (Poisson Process)
            spike_times = []
            neuron_indices = []
            
            for n in range(num_bias_neurons):
                current_time = 0
                while current_time < duration_s:
                    interval = np.random.exponential(1.0/hz)
                    current_time += interval
                    
                    if current_time < duration_s:
                        spike_times.append(int(current_time * 1e6))
                        neuron_indices.append(n)
            
            # 시간순 정렬
            if len(spike_times) > 0:
                sorted_indices = np.argsort(spike_times)
                spike_times = np.array(spike_times)[sorted_indices]
                neuron_indices = np.array(neuron_indices)[sorted_indices]
            
            # 3. 헤더 작성 (13 bytes) - C++ 구조체에 맞춤
            # Label, Data Index는 Bias 데이터에서 중요하지 않다면 0이나 순서대로 기입
            label_id = 0 
            sample_id = i # 파일 내에서의 인덱스
            
            f.write(struct.pack('>B', label_id))   # label (1 byte)
            f.write(struct.pack('>H', sample_id))  # data_index (2 bytes)
            f.write(struct.pack('>I', 0))          # unique_global_id (4 bytes)
            f.write(struct.pack('>I', len(spike_times)))  # num_data_points (4 bytes)
            f.write(struct.pack('>B', 0))          # reserved1 (1 byte)
            f.write(struct.pack('>B', 0))          # reserved2 (1 byte)
            
            # 4. 데이터 작성
            for time, index in zip(spike_times, neuron_indices):
                f.write(struct.pack('>I', int(time)))   # Time (4 bytes)
                f.write(struct.pack('>H', index))       # Neuron Index (2 bytes)

if __name__ == "__main__":
    # --- 설정 파라미터 ---
    HZ = 1           # 10Hz
    DURATION_US = 1000000 
    NUM_NEURONS = 40
    
    TOTAL_SAMPLES = 1000      # 총 필요한 샘플 수 (20클래스 * 300샘플)
    SAMPLES_PER_FILE = 1000   # 파일 하나당 들어갈 샘플 수 (C++ chunk_size에 맞추는 것이 좋음)
    
    OUTPUT_DIR = "dataset_output_chunks"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    NUM_FILES = TOTAL_SAMPLES // SAMPLES_PER_FILE
    
    print(f"Generating {TOTAL_SAMPLES} samples in {NUM_FILES} files ({SAMPLES_PER_FILE} samples per file)...")
    
    for file_idx in range(NUM_FILES):
        # 파일 이름: bias_spikes_0.bin, bias_spikes_1.bin ...
        filename = f"bias_spikes_chunk{file_idx}_{HZ}Hz.bin"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # 시드 오프셋: 파일이 바뀔 때마다 시드 범위가 겹치지 않게 설정
        seed_offset = file_idx * SAMPLES_PER_FILE * 100  
        
        generate_bias_chunk_file(
            output_file=filepath,
            hz=HZ,
            duration_us=DURATION_US,
            num_bias_neurons=NUM_NEURONS,
            num_samples_in_file=SAMPLES_PER_FILE,
            start_seed_offset=seed_offset
        )
        
        print(f"Created {filename} (Samples {file_idx*SAMPLES_PER_FILE} ~ {(file_idx+1)*SAMPLES_PER_FILE - 1})")

    print("\nGeneration Complete!")