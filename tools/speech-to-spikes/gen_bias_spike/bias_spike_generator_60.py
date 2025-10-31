import numpy as np
import struct
import random
import matplotlib.pyplot as plt

def read_bias_spike_file(filename):
    spike_times = []
    neuron_indices = []
    
    with open(filename, 'rb') as f:
        # 헤더 읽기
        label = struct.unpack('>B', f.read(1))[0]
        data_index = struct.unpack('>H', f.read(2))[0]
        unique_global_id = struct.unpack('>I', f.read(4))[0]
        num_data_points = struct.unpack('>I', f.read(4))[0]
        reserved1 = struct.unpack('>B', f.read(1))[0]
        reserved2 = struct.unpack('>B', f.read(1))[0]
        
        # 스파이크 데이터 읽기
        for _ in range(num_data_points):
            time = struct.unpack('>I', f.read(4))[0]
            index = struct.unpack('>H', f.read(2))[0]
            spike_times.append(time)
            neuron_indices.append(index)
    
    return np.array(spike_times), np.array(neuron_indices)

def plot_raster(spike_times, neuron_indices, duration_us, num_neurons, filename="raster_plot.png"):
    plt.figure(figsize=(15, 8))
    
    plt.plot(spike_times/1000000, neuron_indices, '|', color='black', 
            markersize=4, markeredgewidth=1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Poisson Spike Train Raster Plot')
    plt.xlim(0, duration_us/1000000)
    plt.ylim(-1, num_neurons)
    
    plt.yticks(range(0, num_neurons, 5))
    plt.grid(True, which='major', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def generate_bias_spike_train(output_file, hz, duration_us, num_bias_neurons, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    spike_times = []
    neuron_indices = []
    
    # 마이크로초를 초로 변환
    duration_s = duration_us / 1e6
    
    for n in range(num_bias_neurons):
        # 포아송 프로세스로 스파이크 시간 생성
        current_time = 0
        while current_time < duration_s:
            # 다음 스파이크까지의 시간 간격을 지수분포에서 샘플링
            interval = np.random.exponential(1.0/hz)
            current_time += interval
            
            if current_time < duration_s:
                # 마이크로초 단위로 변환하여 저장
                spike_times.append(int(current_time * 1e6))
                neuron_indices.append(n)
    
    # 시간순으로 정렬
    sorted_indices = np.argsort(spike_times)
    spike_times = np.array(spike_times)[sorted_indices]
    neuron_indices = np.array(neuron_indices)[sorted_indices]
    
    # 바이너리 파일 작성
    with open(output_file, 'wb') as f:
        # 헤더 정보
        f.write(struct.pack('>B', 0))  # label
        f.write(struct.pack('>H', 0))  # data_index
        f.write(struct.pack('>I', 0))  # unique_global_id
        f.write(struct.pack('>I', len(spike_times)))  # num_data_points
        f.write(struct.pack('>B', 0))  # reserved1
        f.write(struct.pack('>B', 0))  # reserved2
        
        # 스파이크 데이터 작성
        for time, index in zip(spike_times, neuron_indices):
            f.write(struct.pack('>I', int(time)))
            f.write(struct.pack('>H', index))

if __name__ == "__main__":
    # 기본 파라미터 설정
    hz = 40.0  # 20Hz
    duration_us = 1000000  # 1초
    num_bias_neurons = 60
    random_seed = 42  # 랜덤 시드 설정
    
    # bias 스파이크 파일 생성
    main_file = f"bias_spikes_{int(hz)}Hz_60.bin"
    generate_bias_spike_train(main_file, hz, duration_us, num_bias_neurons, random_seed)
    print(f"Generated bias spike train: {main_file}")

    # 생성된 파일 읽고 시각화하여 파일로 저장
    spike_times, neuron_indices = read_bias_spike_file(main_file)
    png_file = f"bias_raster_plot_{int(hz)}Hz_60.png"
    plot_raster(spike_times, neuron_indices, duration_us, num_bias_neurons, png_file)
