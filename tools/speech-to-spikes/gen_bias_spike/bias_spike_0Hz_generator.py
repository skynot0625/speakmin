import numpy as np
import struct
import random
import matplotlib.pyplot as plt

def generate_bias_spike_train(output_file, hz, duration_us, num_bias_neurons, random_seed=None):
    """
    Bias spike train 생성 함수 - 0Hz인 경우 빈 스파이크 트레인 생성
    """
    # 헤더 정보만 있는 빈 파일 생성
    with open(output_file, 'wb') as f:
        # 헤더 정보
        label = 0
        data_index = 0
        unique_global_id = 0
        num_data_points = 0  # 스파이크 없음
        reserved1 = 0
        reserved2 = 0
        
        # 바이트 순서 변환 (big-endian)
        f.write(struct.pack('>B', label))
        f.write(struct.pack('>H', data_index))
        f.write(struct.pack('>I', unique_global_id))
        f.write(struct.pack('>I', num_data_points))
        f.write(struct.pack('>B', reserved1))
        f.write(struct.pack('>B', reserved2))

# 사용 예시
if __name__ == "__main__":
    hz = 0.0  # 0Hz 설정
    duration_us = 1000000000  # 1000초
    num_bias_neurons = 40
    
    main_file = "bias_spikes_0Hz.bin"
    generate_bias_spike_train(main_file, hz, duration_us, num_bias_neurons)
