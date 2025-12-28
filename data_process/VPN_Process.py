import os
from collections import defaultdict

import orjson
import torch
import numpy as np
from tqdm import tqdm

from pretrain_traffic_process import process_dataset_pcap
from preprocess_utils import load_json


def process_label_files(root_path, file_names):
	flow_list = []
	flow_stats = []
	for file_name in file_names:
		file_path = os.path.join(root_path, file_name)
		flow_data, flow_stat = load_json(file_path, num_flow_packets, packet_len, real_packet_len, need_flow_id=False)
		flow_list.append(flow_data)
		flow_stats.append(flow_stat)
	flow_stats = np.concatenate(flow_stats, axis=0)
	flow_list = np.concatenate(flow_list, axis=0)
	return flow_list, flow_stats


def update_nodeIndex_dic(node_nodeIndex_dic, node_set):
	for node in node_set:
		if node not in node_nodeIndex_dic:
			node_nodeIndex_dic[node] = len(node_nodeIndex_dic)
	return node_nodeIndex_dic


if __name__ == '__main__':
	payload = 256
	packet_len = 128 + payload
	real_packet_len = 32 + payload
	num_flow_packets = 8

	root = ':/data/users/lph/datasets/ISCX_Dataset/ISCX_VPN_2016/VPN'
	save_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ISCX_VPN_2016_OnlyVPN/processed'
	os.makedirs(save_dir, exist_ok=True)

	process_dataset_pcap('ISCX_VPN_2016', payload, num_flow_packets)

	files = list(filter(lambda x: x.endswith('.json'), os.listdir(root)))
	traffic_types = defaultdict(list)

	for file in files:
		traffic_type = file.split('_', 1)[0]
		if traffic_type.isupper():
			traffic_types[traffic_type].append(file)
		else:
			traffic_type = file.split('_', 2)[1]
			if traffic_type == 'p2p':
				traffic_types['P2P'].append(file)
			else:
				traffic_types['AUDIO'].append(file)

	feature_list = []
	label_map = {}
	from concurrent.futures import ProcessPoolExecutor
	with ProcessPoolExecutor(max_workers=len(traffic_types)) as executor:
		for label_index, (label, labeled_files) in enumerate(traffic_types.items()):
			label_map[label_index] = label
			feature = executor.submit(process_label_files, root, labeled_files)
			feature_list.append(feature)

	flow_stat_list = []
	flow_datas = []
	labels = []
	for feature_index, feature in tqdm(enumerate(feature_list), total=len(traffic_types), desc='Process Tor'):
		flow_list, flow_stats = feature.result()

		flow_stat_list.append(flow_stats)
		labels.append(np.full(len(flow_list), feature_index, dtype=np.int8))
		for flow in flow_list:
			if len(flow) < num_flow_packets:
				flow = np.pad(flow, ((0, num_flow_packets - len(flow)), (0, 0)))
			else:
				flow = flow[:num_flow_packets]
			flow_datas.append(flow.reshape(1, num_flow_packets, real_packet_len))

	labels = np.concatenate(labels, axis=0)
	flow_datas = np.vstack(flow_datas)

	flow_stats = np.concatenate(flow_stat_list, axis=0)
	flow_stats = np.asarray(flow_stats)
	min_packet_num = np.min(flow_stats)
	max_packet_num = np.max(flow_stats)
	avg_packet_num = np.average(flow_stats)
	median_packet_num = np.median(flow_stats)
	print(f'Packet Stat: Min: {min_packet_num}, Max: {max_packet_num}, Avg: {avg_packet_num}, Median: {median_packet_num}')

	assert (len(labels) == len(flow_datas))
	print(f'Total {len(flow_datas)} Flows.')

	# torch.save(adj, f'{save_dir}/adj.pt')
	np.save(f'{save_dir}/stream_feat.npy', flow_datas)
	np.save(f'{save_dir}/label.npy', labels)
	torch.save(label_map, f'{save_dir}/label_map.pt')
