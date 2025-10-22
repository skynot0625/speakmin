# Copyright contributors to the speakmin project
# SPDX-License-Identifier: Apache-2.0
#
# Python class for Google Speech Command Dataset (GSCD)
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import pickle
import random
import multiprocessing
from .audio_core import AudioCore
import gc
from tqdm import tqdm  # 프로그레스 바를 위한 라이브러리 추가


class GSCD(object):
    def __init__(self, dataset_path):
        """Initialize GSCD with dataset path"""
        assert os.path.isdir(dataset_path)
        self.dataset_path = dataset_path
        self.categories = [os.path.basename(path) for path in glob.glob(dataset_path + '/*') 
                         if os.path.isdir(path)]
        self.wav_files = []
        self.uid = {}
        uid = 0  # unique id for each .wav file
        
        for category in self.categories:
            wav_files = glob.glob(os.path.join(dataset_path, category, '*.wav'))
            for wav_file in wav_files:
                self.wav_files.append({
                    'uid': uid,
                    'category': category,
                    'wav_file': wav_file
                })
                self.uid[wav_file] = uid
                uid += 1
                
        self.queue = multiprocessing.Manager().Queue()
        self.result_queue = multiprocessing.Manager().Queue()

    def _get_wav_file_names_from_file(self, list_file):
        """Get wav file names from list file"""
        with open(list_file, 'r') as f:
            list_of_dict = []
            for s_line in f:
                category, wav_file = s_line.strip().split('/')
                wav_file_full_path = os.path.join(self.dataset_path, category, wav_file)
                uid = self.uid[wav_file_full_path]
                list_of_dict.append({
                    'uid': uid,
                    'category': category,
                    'wav_file': wav_file_full_path
                })
            categories = set(d.get('category') for d in list_of_dict)
            return list_of_dict, categories

    @classmethod
    def category2index(cls, category):
        """Convert category name to index"""
        category_list = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
            'cat', 'dog', 'bird', 'house', 'bed', 'tree', 'happy', 'wow', 'sheila', 'marvin',
            'forward', 'backward', 'follow', 'learn', 'visual', '_background_noise_'
        ]
        assert category in category_list
        return category_list.index(category)

    def show_categories(self):
        """Show available categories and their file counts"""
        for category in self.categories:
            print(f'{category}: {len([x for x in self.wav_files if x["category"] == category])}')

    def _exclude_wav_file(self, list_of_dict, exclude_list_of_dict, dict_key='wav_file'):
        """Exclude wav files from list"""
        exclude_wav_file_set = set(dict_data[dict_key] for dict_data in exclude_list_of_dict)
        return [dict_data for dict_data in list_of_dict 
                if dict_data[dict_key] not in exclude_wav_file_set]

    def _select_wav_file_source(self, wav_file_source):
        """Select wav file source based on specified criteria"""
        assert wav_file_source in ['all', 'testing', 'validation', 'not_testing', 
                                 'not_validation', 'not_testing_validation']
        
        if wav_file_source == 'all':
            return self.wav_files, self.categories
            
        if wav_file_source == 'testing':
            return self._get_wav_file_names_from_file(
                os.path.join(self.dataset_path, 'testing_list.txt'))
                
        if wav_file_source == 'validation':
            return self._get_wav_file_names_from_file(
                os.path.join(self.dataset_path, 'validation_list.txt'))
                
        if wav_file_source == 'not_testing':
            tst_files, _ = self._get_wav_file_names_from_file(
                os.path.join(self.dataset_path, 'testing_list.txt'))
            source_wav_files = self._exclude_wav_file(self.wav_files, tst_files)
            return source_wav_files, set(d.get('category') for d in source_wav_files)
            
        if wav_file_source == 'not_validation':
            val_files, _ = self._get_wav_file_names_from_file(
                os.path.join(self.dataset_path, 'validation_list.txt'))
            source_wav_files = self._exclude_wav_file(self.wav_files, val_files)
            return source_wav_files, set(d.get('category') for d in source_wav_files)
            
        # not_testing_validation
        tst_files, _ = self._get_wav_file_names_from_file(
            os.path.join(self.dataset_path, 'testing_list.txt'))
        val_files, _ = self._get_wav_file_names_from_file(
            os.path.join(self.dataset_path, 'validation_list.txt'))
        not_tst_files = self._exclude_wav_file(self.wav_files, tst_files)
        source_wav_files = self._exclude_wav_file(not_tst_files, val_files)
        return source_wav_files, set(d.get('category') for d in source_wav_files)

    def random_split(self, category_list, split_number_list, rng=None, rng_seed=10, wav_file_source='all', use_all_in_source=False):
        source_wav_files, source_categories = self._select_wav_file_source(wav_file_source)
        print(f"Processing categories: {category_list}")
        print(f"Split numbers: {split_number_list}")
        
        assert all(category in source_categories for category in category_list)
        
        if rng is None:
            rng = np.random.RandomState(rng_seed)
            
        splitted_wav_files = []
        
        # Process each category
        for category in category_list:
            wav_files = [d for d in source_wav_files if d['category'] == category]
            print(f"Processing {category}: Found {len(wav_files)} files")
            
            if use_all_in_source:
                current_split = len(wav_files)
            else:
                current_split = split_number_list[0]  # Use first split number
                
            assert current_split <= len(wav_files), \
                f"Not enough files for {category}. Need: {current_split}, Have: {len(wav_files)}"
            
            # Shuffle files
            wav_files_shuffled = rng.permutation(wav_files).tolist()
            selected_files = wav_files_shuffled[:current_split]
            
            # Add metadata - split files into multiple parts
            num_splits = len(split_number_list)
            files_per_split = current_split // num_splits
            
            for split_index in range(num_splits):
                start_idx = split_index * files_per_split
                end_idx = start_idx + files_per_split
                
                for data_index, dict_data in enumerate(selected_files[start_idx:end_idx]):
                    dict_data['split_index'] = split_index
                    dict_data['data_index'] = data_index
                    dict_data['label'] = self.category2index(category)
                    splitted_wav_files.append(dict_data)
        
        print(f"Total processed files: {len(splitted_wav_files)}")
        return splitted_wav_files

    def split_and_shuffle_for_dump(self, spikes_list_of_dict, shuffle=True, rand_seed=10):
        """Split and shuffle spikes data for dumping"""
        # Get all unique split indices
        split_indices = sorted(set(d['split_index'] for d in spikes_list_of_dict))
        print(f"Processing {len(split_indices)} splits")
        
        result = []
        for split_index in split_indices:
            # Get data for current split
            split_data = [d for d in spikes_list_of_dict if d['split_index'] == split_index]
            
            if shuffle:
                random.seed(rand_seed)
                random.shuffle(split_data)
                
            result.append(split_data)
            print(f"Split {split_index}: {len(split_data)} files")
            
        return result

    def convert2spikes(self, splitted_wav_files, n_mels, vth, **kwargs):
        """Convert audio files to spikes with memory efficient processing"""
        chunk_size = kwargs.pop('chunk_size', 1000)
        num_workers = kwargs.get('num_process', 4)
        results = []
        
        # Group files by category to maintain balance
        categories = set(d['category'] for d in splitted_wav_files)
        for category in categories:
            category_files = [d for d in splitted_wav_files if d['category'] == category]
            category_results = []
            
            # Process each category in chunks
            for i in range(0, len(category_files), chunk_size):
                chunk = category_files[i:i + chunk_size]
                self._put_items_in_queue(chunk, n_mels, vth, **kwargs)
                self._start_multi_process(num_workers)
                chunk_results = self._get_results_from_result_queue()
                category_results.extend(chunk_results)
                
                # Clean up
                del chunk
                del chunk_results
            
            results.extend(category_results)
            del category_results
        
        return results

    def _put_items_in_queue(self, splitted_wav_files, n_mels, vth, **kwargs):
        for dict_data in splitted_wav_files:
            params = {
                'n_mels': n_mels,
                'vth': vth,
                **kwargs,
                **dict_data
            }
            self.queue.put(params)
            
        for _ in range(kwargs.get('num_process', 4)):
            self.queue.put(None)

    def _start_multi_process(self, num_process):
        processes = []
        for _ in range(num_process):
            p = multiprocessing.Process(
                target=self._worker_queue,
                args=(self.queue, self.result_queue)
            )
            p.daemon = True
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def _get_results_from_result_queue(self):
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results

    @classmethod
    def _worker_queue(cls, queue, result_queue):
        while True:
            try:
                item = queue.get()
                if item is None:
                    break

                sound = AudioCore(item['wav_file'])
                if item.get('preemphasis'):
                    sound.preemphasis(coef=item['preemphasis_coef'])
                if item.get('norm'):
                    sound.normalize(target_dBFS=item['norm_target_dBFS'])
                if item.get('align'):
                    sound.align_sound(
                        index_align=8000,
                        n_points=16000,
                        pad_zero=item['pad_zero']
                    )

                spikes = sound.speech2spikes(
                    item['n_mels'],
                    item['vth'],
                    alpha=item['alpha'],
                    time_unit=item['time_unit'],
                    norm=item['mel_norm'],
                    vcsv_file_list=item['vcsv_file_list'],
                    leak_enable=item['leak_enable'],
                    leak_tau=item['leak_tau']
                )

                result_queue.put({
                    **item,
                    'num_points': len(spikes),
                    'spikes': spikes
                })

            finally:
                # Clean up memory
                if 'sound' in locals():
                    del sound
                if 'spikes' in locals():
                    del spikes
                if 'item' in locals():
                    del item

    def dump_as_binary(self, spikes_list_of_dict, output_file):
        # dump as binary data
        # Header: BHIBB
        # Contents: (IH) * (number of spikes)
        with open(output_file, 'wb') as f:
            for data in spikes_list_of_dict:
                # struct.pack format
                # B: unsigned char,  1 byte  (8 bit)
                # H: unsigned short, 2 bytes (16 bit)
                # I: unsigned int,   4 bytes (32 bits)
                # endian
                # <: little endian
                # >: big endian
                header = struct.pack(
                    '>BHIIBB',
                    data['label'],
                    data['data_index'],
                    data['uid'],
                    data['num_points'],
                    0, 0
                )
                f.write(header)
                
                for spike in data['spikes']:
                    f.write(struct.pack('>IH', spike[0], spike[1]))

    def dump_as_pickle(self, spikes_list_of_dict, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(spikes_list_of_dict, f)
    #
    # Plot utilities
    #----------------------------
    @classmethod
    def raster_plots(cls, spikes_list_of_dict, data_index_list = list(range(8)), plot_title = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'
        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])

        # use the first data assuming all data has same parameter value
        n_mels = spikes_list_of_dict[0]['n_mels']
        align = spikes_list_of_dict[0]['align']
        time_unit = spikes_list_of_dict[0]['time_unit']
        norm = spikes_list_of_dict[0]['norm']
        preemphasis = spikes_list_of_dict[0]['preemphasis']

        file_ext_norm = '_norm' if norm else ''
        file_ext_align = '_align' if align else ''
        file_ext_preemphasis = '_preemphasis' if preemphasis else ''
        file_ext = f'_raster{file_ext_preemphasis}{file_ext_norm}{file_ext_align}.png'
        for category in categories:
            ofile = category + file_ext
            sub_figsize = [4, 3]
            fig_col = 4
            fig_row = int((len(data_index_list) + fig_col - 1) / fig_col)
            figsize = [sub_figsize[0] * fig_col, sub_figsize[1] * fig_row]
            fig, ax = plt.subplots(fig_row, fig_col, figsize = figsize)

            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]

            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            assert all([index in [dict_data['data_index'] for dict_data in sorted_target_list_of_dict] for index in data_index_list])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                row, col = index // fig_col, index % fig_col
                x_list = [spike[0] * time_unit for spike in dict_data['spikes']]
                y_list = [spike[1] for spike in dict_data['spikes']]
                ax_target = ax[row, col] if fig_col < len(data_index_list) else ax[col]
                ax_target.plot(x_list, y_list, '|b')
                ax_target.set_ylim(-0.5, n_mels - 0.5)
                ax_target.set_xlim(-0.05, 1.05)
                ax_target.set_xlabel('Time [s]')
                ax_target.set_ylabel('Cnannel#')
                if plot_title:
                    ax_target.set_title(f'"{category}",{data_index},{os.path.basename(dict_data["wav_file"])}')
            fig.tight_layout()
            fig.savefig(ofile)
            plt.close(fig)

    @classmethod
    def wform_lookup_ylim(cls, spikes_list_of_dict, data_index_list = list(range(8)), plot_original = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'

        # use the first data assuming all data has same parameter value
        norm = False if plot_original else spikes_list_of_dict[0]['norm']
        norm_target_dBFS = spikes_list_of_dict[0]['norm_target_dBFS']
        preemphasis = False if plot_original else spikes_list_of_dict[0]['preemphasis']
        preemphasis_coef = spikes_list_of_dict[0]['preemphasis_coef']

        max_value = 0
        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])
        for category in categories:
            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]
            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                sound = AudioCore(dict_data['wav_file'])
                if preemphasis:
                    sound.preemphasis(coef = preemphasis_coef)
                if norm:
                    sound.normalize(target_dBFS = norm_target_dBFS)
                sig_max = max(max(sound.sig), abs(min(sound.sig)))
                if sig_max > max_value:
                    max_value = sig_max
        return max_value

    @classmethod
    def wform_plots(cls, spikes_list_of_dict, data_index_list = list(range(8)),
                    ylim = [-30000, 30000], xlim = [-0.05, 1.05], plot_title = False, plot_original = False):
        assert len(set([spikes_dict['split_index'] for spikes_dict in spikes_list_of_dict])) == 1, \
            f'split_index should be all same.'

        # use the first data assuming all data has same parameter value
        if plot_original:
            align, pad_zero, norm, preemphasis = [False, False, False, False]
        else:
            align = spikes_list_of_dict[0]['align']
            pad_zero = spikes_list_of_dict[0]['pad_zero']
            norm = spikes_list_of_dict[0]['norm']
            preemphasis = spikes_list_of_dict[0]['preemphasis']
        norm_target_dBFS = spikes_list_of_dict[0]['norm_target_dBFS']
        preemphasis_coef = spikes_list_of_dict[0]['preemphasis_coef']

        file_ext_norm = '_norm' if norm else ''
        file_ext_align = '_align' if align else ''
        file_ext_preemphasis = '_preemphasis' if preemphasis else ''
        file_ext = f'_wform{file_ext_preemphasis}{file_ext_norm}{file_ext_align}.png'

        categories = set([dict_data['category'] for dict_data in spikes_list_of_dict])
        for category in categories:
            ofile = category + file_ext
            sub_figsize = [4, 3]
            fig_col = 4
            fig_row = int((len(data_index_list) + fig_col - 1) / fig_col)
            figsize = [sub_figsize[0] * fig_col, sub_figsize[1] * fig_row]
            fig, ax = plt.subplots(fig_row, fig_col, figsize = figsize)
            target_list_of_dict = [dict_data for dict_data in spikes_list_of_dict if dict_data['category'] == category]

            sorted_target_list_of_dict = sorted(target_list_of_dict, key=lambda x:x['data_index'])
            assert all([index in [dict_data['data_index'] for dict_data in sorted_target_list_of_dict] for index in data_index_list])
            for index, data_index in enumerate(data_index_list):
                dict_data_list = [dict_tmp for dict_tmp in sorted_target_list_of_dict if dict_tmp['data_index'] == data_index]
                assert len(dict_data_list) == 1
                dict_data = dict_data_list[0]
                sound = AudioCore(dict_data['wav_file'])
                if preemphasis:
                    sound.preemphasis(coef = preemphasis_coef)
                if norm:
                    sound.normalize(target_dBFS = norm_target_dBFS)
                if align:
                    sound.align_sound(index_align = 8000, n_points = 16000, pad_zero = pad_zero)
                row, col = index // fig_col, index % fig_col
                ax_target = ax[row, col] if fig_col < len(data_index_list) else ax[col]
                sound.plot_original(ax_target)
                ax_target.set_ylim(ylim[0], ylim[1])
                ax_target.set_xlim(xlim[0], xlim[1])
                ax_target.set_xlabel('Time [s]')
                if plot_title:
                    ax_target.set_title(f'"{category}",{data_index},{os.path.basename(dict_data["wav_file"])}')
            fig.tight_layout()
            fig.savefig(ofile)
            plt.close(fig)
