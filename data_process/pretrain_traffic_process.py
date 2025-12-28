import os
from os.path import exists
from typing import Union, Sequence

import torch

from preprocess_utils import load_jsons, show_flow_stat


def parse_pcap_to_json(pcap_file_path: str, goal_path: str,
					   payload_size: int = 100, num_window_packets: int = 5,
					   hd_dead_path: str = None)\
		-> Union[str, None]:
	"""
	调用capture工具, 将.pcap或.pcapng文件转换为.json文件
	:param pcap_file_path:      type=str, .pcap或.pcapng文件路径
	:param batch_index:         Optional, type=str, 批次索引, default=None
	:param goal_file_name:      type=str, 生成的目标文件名
	:param payload_size:        type=int, 要生成的json文件中, 每个数据包包含的payload字节数
	:param num_window_packets:  type=int, 每条流最小数据包数
	:param use_uint32:          type=bool, 是否使用uint32表示数据包内数据, 否则使用uint8, default=False
	:return:                    type=str, 返回生成的.json文件路径
	"""

	# --stride=8表示使用int8, -U表示使用无符号整数, 即使用uint8,
	# -tTu4分别表示 使用TCP头、时间戳、UDP头和IPv4头, -Z 0 表示若不存在则填充0
	# -p {payload_size}表示获取Payload中前payload_size字节长度数据
	# -L {num_window_packets }表示一条网络流中最少数据包数为num_window_packets, 否则舍弃
	# -R 100 表示一条网络流中最大数据包数为100, 否则分割
	# -P -W 分别表示要处理的pcap文件路径和输出文件路径
	# command = f"capture --stride={stride} -tTu4 -p{payload_size} -L{num_window_packets} -R100 -U -Z 0 -P{pcap_file_path} -W{goal_path}"
	if hd_dead_path is None:
		hd_dead_path = '/home/yhy/project/flow_test/hd-dead'
	command = f"{hd_dead_path} -S8 -UCTV -p{payload_size} -L{num_window_packets} -R100 -P{pcap_file_path} -W{goal_path} -J8 --filter=\"(ip or vlan) and (not (net 224.0.0.0/3 or host 255.255.255.255 or host 0.0.0.0))\""
	return do_command(command, goal_path)


def do_command(command: str, goal_path: str = None) -> Union[str, None]:
	import subprocess
	print(f'Executing command: {command}')
	try:
		result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		print(result.stdout)
		return goal_path
	except subprocess.CalledProcessError as e:
		print("命令执行出错:", e)
	except Exception as e:
		print("发生了异常:", e)


def process_pretrain_pcap(payload_size, num_window_packets, hd_dead_path=None):
	root_path = '/home/yhy/project/flow_test/data/benign'
	save_path = '../data/pretrain/raw'
	os.makedirs(save_path, exist_ok=True)
	data_dirs = list(filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path)))
	for dir_name in data_dirs:
		data_dir = os.path.join(root_path, dir_name, 'raw')
		for batch_index in range(24):
			file_path_prefix = os.path.join(data_dir, f'capture_{batch_index}')
			pcap_path = f'{file_path_prefix}.pcapng'
			if not exists(pcap_path):
				pcap_path = f'{file_path_prefix}.pcap'
			goal_path = f'{save_path}/{dir_name}_{batch_index}_p{payload_size}_w{num_window_packets}.json'
			parse_pcap_to_json(pcap_path, goal_path, payload_size, num_window_packets, hd_dead_path)


def process_dataset_pcap(dataset_name, payload_size, num_window_packets, hd_dead_path=None):
	# dataset_path = f'/home/yhy/project/flow_test/data/{dataset_name}/raw'
	dataset_path = f'/data/users/lph/datasets/ISCX_Dataset/ISCX_VPN_2016/VPN'
	print(f'Start Process {dataset_path}')
	save_path = f'/data/users/lph/datasets/ISCX_Dataset/ISCX_VPN_2016/VPN/raw'
	os.makedirs(save_path, exist_ok=True)
	files = list(filter(lambda x: x.endswith('.pcap') or x.endswith('.pcapng'), os.listdir(dataset_path)))
	for file_name in files:
		pcap_path = os.path.join(dataset_path, file_name)
		goal_path = f'{save_path}/{file_name}_p{payload_size}_w{num_window_packets}.json'
		if not exists(goal_path):
			parse_pcap_to_json(pcap_path, goal_path, payload_size, num_window_packets, hd_dead_path)


def process_json(dataset_name: str = None, payload_size: int = 96,
				 num_window_packets: int = 8,
				 root_path: str = None, save_path: Union[str, Sequence[str]] = None,
				 need_flow_id: bool = False, ip_is_node: bool = False):
	assert dataset_name is not None or (root_path is not None and save_path is not None)
	if root_path is None:
		root_path = f'../data/{dataset_name}/raw'
	if save_path is None:
		save_path = f'../data/{dataset_name}/processed/{dataset_name}_data_p{payload_size}_w{num_window_packets}.pt'
	if need_flow_id:
		assert (not isinstance(save_path, str) and len(save_path) == 3)
		save_dir = save_path[0].rsplit('/', 1)[0]
	else:
		save_dir = save_path.rsplit('/', 1)[0]
	os.makedirs(save_dir, exist_ok=True)
	json_files = sorted(os.listdir(root_path))
	json_paths = [os.path.join(root_path, file_name) for file_name in json_files]

	raw_packet_len = payload_size + 128
	real_packet_len = 32 + payload_size
	processed_datas = load_jsons(json_paths, num_window_packets, raw_packet_len, real_packet_len, need_flow_id, ip_is_node)
	if need_flow_id:
		flow_list, flow_id_list, flow_stats, node_sets = processed_datas
		torch.save(flow_id_list, save_path[1])
		torch.save(node_sets, save_path[2])
	else:
		flow_list, flow_stats = processed_datas
	flow_list = torch.as_tensor(flow_list, dtype=torch.uint8)
	show_flow_stat(flow_stats)
	if need_flow_id:
		torch.save(flow_list, save_path[0])
	else:
		torch.save(flow_list, save_path)


if __name__ == '__main__':
	# payload_size = 32 * 3
	payload_size = 96
	packet_len = 128 + payload_size
	real_packet_len = 32 + payload_size
	num_window_packets = 8
	# hd_dead_path = None
	hd_dead_path = '/data/yhy/hd-dead'

	for dataset_name in [
		'ISCX-2012',
		'CIC-IDS-2017',
		# 'pretrain',
		# 'ISCX-Bot-2014'
	]:
		if dataset_name == 'pretrain':
			process_pretrain_pcap(payload_size, num_window_packets, hd_dead_path)
		else:
			process_dataset_pcap(dataset_name, payload_size, num_window_packets, hd_dead_path)
		process_json(dataset_name, payload_size, num_window_packets)

