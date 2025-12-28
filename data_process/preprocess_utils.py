import os.path
import shutil

import numpy as np
import orjson
from scipy.sparse import csr_matrix
from tqdm import tqdm


def delete_dir(folder_path):
	""" 删除指定的文件夹及其内容 """
	try:
		shutil.rmtree(folder_path)
		print(f"Folder '{folder_path}' and its contents have been deleted.")
	except FileNotFoundError:
		print(f"Folder '{folder_path}' does not exist.")
	except PermissionError:
		print(f"Permission denied: Unable to delete folder '{folder_path}'.")
	except Exception as e:
		print(f"An error occurred: {e}")


# 将边集(dtype=str)转换为带编号的边集(多重边图), 去除自环
def remove_self_loops_and_renumber_edges(edges, edge_feat=None, edge_label=None):
	node_set = set(edges.ravel())
	num_nodes = len(node_set)

	node_nodeIndex_dic = {node: node_index for node_index, node in enumerate(node_set)}
	new_edges = np.empty(shape=(len(edges), 2), dtype=np.int32)
	# 根据节点索引和邻接矩阵, 初步构造边集(多重边图), 去除自环
	self_loops = []
	for edge_index, (u, v) in enumerate(edges):
		if u == v:
			self_loops.append(edge_index)
		else:
			new_edges[edge_index][0] = node_nodeIndex_dic[u]
			new_edges[edge_index][1] = node_nodeIndex_dic[v]
	others = []
	if len(self_loops) > 0:
		new_edges = np.delete(new_edges, self_loops, axis=0)
		if edge_feat is not None:
			edge_feat = np.delete(edge_feat, self_loops, axis=0)
			others.append(edge_feat)
		if edge_label is not None:
			edge_label = np.delete(edge_label, self_loops, axis=0)
			others.append(edge_label)
	else:
		if edge_feat is not None:
			others.append(edge_feat)
		if edge_label is not None:
			others.append(edge_label)
	print(f'Removed {len(self_loops)} Self Loops.')
	return new_edges, num_nodes, *others


def process_multiple_graph_to_simple_graph(edges, edge_feat, edge_label, num_nodes: int):
	# 根据边集和标签, 构造简单图边集
	edge_label_dic = dict()
	for edge_index, (u, v) in enumerate(edges):
		edge = (u, v)
		if edge not in edge_label_dic:
			edge_label_dic[edge] = edge_index
		elif edge_label[edge_label_dic[edge]] == 0 and edge_label[edge_index] != 0:
			edge_label_dic[edge] = edge_index

	selected_edges = np.empty(shape=len(edge_label_dic), dtype=np.int32)
	for index, edge_index in enumerate(edge_label_dic.values()):
		selected_edges[index] = edge_index
	edges = edges[selected_edges]
	if edge_feat is not None:
		edge_feat = edge_feat[selected_edges]
	edge_label = edge_label[selected_edges].astype(np.int8)
	adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes))
	return selected_edges, edges, edge_feat, edge_label, adj


def get_node_edge_dic(edges):
	from collections import defaultdict
	node_edge_dic = defaultdict(list)
	for edge_index, (u, v) in enumerate(edges):
		node_edge_dic[u].append(edge_index)
		node_edge_dic[v].append(edge_index)
	node_edge_dic = {node: np.asarray(correlated_edges, dtype=np.int32) for node, correlated_edges in node_edge_dic.items()}
	return node_edge_dic


def load_json(path, num_window_packets, packet_len, real_packet_len, need_flow_id: bool = False, ip_is_node: bool = False):
	flow_datas, flow_stats = [], []
	if need_flow_id:
		flow_ids, node_set = [], set()
	with open(path, "r") as json_file:
		for line_index, line in tqdm(enumerate(json_file), desc=f'Process {path.rsplit("/")[-1]}'):
			if line_index == 0:
				continue
			else:
				# 去掉,或者]
				line = line[:-2]
			flow_data = orjson.loads(line)
			if need_flow_id:
				protocol = flow_data['flowId'].rsplit('_', 1)[-1]
				first_packet_data = np.fromstring(string=flow_data['data'][0]['bitvec'], dtype=np.uint8, count=packet_len, sep=',')
				src_ip = '.'.join(map(str, first_packet_data[12:16]))
				dst_ip = '.'.join(map(str, first_packet_data[16:20]))
				if ip_is_node:
					src_node, dst_node = src_ip, dst_ip
				else:
					if protocol == 'TCP':
						src_port = int.from_bytes(first_packet_data[60:62], byteorder='big')
						dst_port = int.from_bytes(first_packet_data[62:64], byteorder='big')
					else:
						src_port = int.from_bytes(first_packet_data[120:122], byteorder='big')
						dst_port = int.from_bytes(first_packet_data[122:124], byteorder='big')
					src_node, dst_node = f'{src_ip}_{src_port}', f'{dst_ip}_{dst_port}'
				flow_ids.append(np.asarray([src_node, dst_node]))
				node_set.add(src_node)
				node_set.add(dst_node)

			flow_stats.append(flow_data['count'])
			flow_data = flow_data['data'][:num_window_packets]
			processed_flow = np.empty((num_window_packets, real_packet_len), dtype=np.uint8)
			for packet_index, packet_data in enumerate(flow_data):
				packet_data = np.fromstring(string=packet_data['bitvec'], dtype=np.uint8, count=packet_len, sep=',')
				processed_flow[packet_index, :12] = packet_data[:12]
				processed_flow[packet_index, 12: 28] = packet_data[64: 80]
				processed_flow[packet_index, 28:] = packet_data[124:]
			flow_datas.append(processed_flow)

	flow_stats = np.asarray(flow_stats)
	flow_datas = np.stack(flow_datas)
	if need_flow_id:
		flow_ids = np.stack(flow_ids)
		return flow_datas, flow_ids, flow_stats, node_set
	return flow_datas, flow_stats


def load_jsons(paths, num_window_packets, packet_len, real_packet_len, need_flow_id: bool = False, ip_is_node: bool = False):
	flow_list, flow_stats = [], []
	if need_flow_id:
		flow_id_list, node_sets = [], set()
	json_process_progress = tqdm(paths, total=len(paths))
	for path in json_process_progress:
		# 忽略空文件
		if os.path.getsize(path) == 2:
			continue
		json_process_progress.set_description(f'Process {path.rsplit("/", 1)[-1]}')
		try:
			json_data = load_json(path, num_window_packets, packet_len, real_packet_len, need_flow_id=need_flow_id, ip_is_node=ip_is_node)
		except UnicodeDecodeError as e:
			print(path)
			print(e)
			exit()
		if need_flow_id:
			flow_data, flow_id, flow_stat, node_set = json_data
			flow_id_list.append(flow_id)
			node_sets = node_set.union(node_set)
		else:
			flow_data, flow_stat = json_data
		flow_list.append(flow_data)
		flow_stats.append(flow_stat)
	flow_list = np.concatenate(flow_list, axis=0)
	flow_stats = np.concatenate(flow_stats, axis=0)
	if need_flow_id:
		flow_id_list = np.concatenate(flow_id_list, axis=0)
		return flow_list, flow_id_list, flow_stats, node_sets
	return flow_list, flow_stats


def show_flow_stat(flow_stats):
	min_packet_num = np.min(flow_stats)
	max_packet_num = np.max(flow_stats)
	avg_packet_num = np.average(flow_stats)
	median_packet_num = np.median(flow_stats)
	print(f'\nPacket Stat: Min: {min_packet_num}, Max: {max_packet_num}, Avg: {avg_packet_num}, Median: {median_packet_num}\n')


if __name__ == '__main__':
	pass
