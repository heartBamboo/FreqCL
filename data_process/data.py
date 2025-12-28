import os 
import shutil
import subprocess

import binascii
import scapy.all as scapy
import numpy as np
import torch
import tqdm
from scapy.error import Scapy_Exception
from torch.utils.data import DataLoader, TensorDataset

ignored_ips = {'0.0.0.0', '255.255.255.255'}
# 在Windows下SplitCap的分割IP为192-168-0-1, Linux下为192.168.0.1
for ip in list(ignored_ips):
    ignored_ips.add(ip.replace('.', '-', 3))


def ignored_ip(ips):
    return any(ip in ignored_ips or ip.startswith('224') for ip in ips)
    # return any(ip in ignored_ips for ip in ips)


def process_ip(pcap_file_path, origin_pcap_file_name):
    pcap_file_name = pcap_file_path.rsplit('/', 1)[-1]
    if ':' in pcap_file_name:
        return None
    # test.pcap.TCP_13-35-210-49_443_172-26-248-34_63662.pcap
    _, ip1, _, ip2, _ = pcap_file_name.split(origin_pcap_file_name, 1)[-1].split('_')
    if ignored_ip([ip1, ip2]):
        return None
    return ip1, ip2


def split_pcap(input_file: str, output_dir: str, num_flow_packets: int):
    """按照五元组将pcap文件分割成网络流

    Args:
        input_file (str): pcap文件路径
        output_dir (str): 分割后的网络流存放路径
        num_flow_packets (int): 网络流中的最大包数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    command = f'mono /data/users/lph/tools/SplitCap.exe -r {input_file} -o {output_dir} -d -s session -p 100000'
    # command = f'~/splitter -i {input_file} -o {output_dir} -l {num_flow_packets} -f five_tuple'
    subprocess.run(command, shell=True)


def find_files(data_path, extension=".pcap"):
    """查找目录下的所有pcap文件

    Args:
        data_path (_type_): 目录
        extension (str, optional): _description_. Defaults to ".pcap".

    Returns: 所有pcap文件的路径列表
    """
    pcap_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files


def packet_to_array(packet, header_length=160, payload_length=104):
    """将scapy库解析得到的数据包转换为字节数组

    Args:
        packet (_type_): scapy库解析得到的数据包
        header_length (int, optional): 数据包头部的长度(16进制数的个数). Defaults to 160.
        payload_length (int, optional): 数据包负载的长度(16进制数的个数). Defaults to 104.

    Returns: 
        packet_data (np.ndarray): 数据包对应的字节数组，长度应为去掉IP和端口号的数据包头部长度+负载长度=(header_length // 2 - 12) + (payload_length // 2), 单位是字节
    """
    # 从数据包中解析IP数据报的头部信息
    header = (binascii.hexlify(bytes(packet["IP"]))).decode()
    # 把负载信息从头部去除
    try:
        payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
        header = header.replace(payload, '')
    except:
        payload = None
    
    # 数据包头部的长度上去除IP头和端口号，单位转换为字节
    real_header_length = header_length // 2 - 12
    # 算数据包的长度，单位转换为字节
    real_packet_length = real_header_length + payload_length // 2
    
    # 将解析得到的数据包头部转换成字节数组，并去掉IP头和端口号
    header = string_to_array(header[:header_length])
    header = np.concatenate((header[:12], header[24:]))
    
    # 如果没有负载，则填充0
    if payload is None:
        packet_data = np.pad(header, (0, real_packet_length-len(header)))
    else:
        payload = string_to_array(payload[:payload_length])
        # 如果长度无需填充则直接拼接返回
        if len(header) == real_header_length and len(payload) == payload_length // 2:
            packet_data = np.concatenate((header, payload))
        else:
        # 隐式填充: 构造全0数组再将有效值赋值到对应位置, 相比pad减少了临时数组数量
            packet_data = np.zeros(real_packet_length, dtype=np.uint8)
            packet_data[:len(header)] = header
            packet_data[real_header_length: real_header_length+len(payload)] = payload
    return packet_data
    

# 将字符串转换为数组
def string_to_array(flow_string):
    return np.asarray([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)], dtype=np.uint8)


def process_pcap_to_array(pcap_filename, num_flow_packets=5, header_length=160, payload_length=104):
    """将pcap文件转换为数组

    Args:
        pcap_filename (_type_): _description_
        num_flow_packets (int, optional): _description_. Defaults to 5.

    Returns: 网络流字节数组
    """
    real_packet_length = (header_length // 2 - 12) + (payload_length // 2)
    flow_length = real_packet_length * num_flow_packets

    # 解析pcap文件，保留前num_flow_packets个包
    packets = scapy.rdpcap(pcap_filename)[:num_flow_packets]
    flow_data = []
    for packet_index, packet in enumerate(packets):
        # 遍历packets，将每个packet转成字节数组
        try:
            packet_data = packet_to_array(packet, header_length, payload_length)
        except:
            # 解析过程中，如果第一个包解析失败，舍弃整个网络流。
            if packet_index == 0:
                return None
            # 解析过程中，如果后续的包解析失败，使用0填充该包对应的数据。
            packet_data = np.zeros(flow_length // num_flow_packets, dtype=np.uint8)
        flow_data.append(packet_data)
    # 将最多num_flow_packets个包拼接成对应的网络流数据
    flow_data = np.concatenate(flow_data)
    # 如果网络流数据的长度不足flow_length，使用0填充。
    if len(flow_data) < flow_length:
        flow_data = np.pad(flow_data, (0, flow_length-len(flow_data)))
    return flow_data
    
    
def process_ISCX_Bot_dataset(dataset_dir: str, output_dir: str, num_flow_packets:int = 5, header_length: int = 160, payload_length: int = 104):
    """处理ISCX-Bot-2014数据集

    Args:
        dataset_dir (str): 原始数据集路径
        output_dir (str): 输出处理数据的文件夹路径
        num_flow_packets (int, optional): 网络流中最大包个数. Defaults to 5.
        header_length (int, optional): 数据包头部长度, 单位16进制数个数. Defaults to 160.
        payload_length (int, optional): 数据包负载长度, 单位16进制数个数. Defaults to 104.
    """
    malicious_ips = {
        '192.168.2.112', '131.202.243.84', '192.168.5.122', '198.164.30.2', '192.168.2.110', '192.168.4.118',
        '192.168.2.113', '192.168.1.103', '192.168.4.120', '192.168.2.109', '192.168.2.105', '147.32.84.180',
        '147.32.84.170', '147.32.84.150', '147.32.84.140', '147.32.84.130', '147.32.84.160', '10.0.2.15',
        '192.168.106.141', '192.168.106.131', '172.16.253.130', '172.16.253.131', '172.16.253.129', '172.16.253.240',
        '74.78.117.238', '158.65.110.24', '192.168.3.35', '192.168.3.25', '192.168.3.65', '172.29.0.116',
        '172.29.0.109', '172.16.253.132', '192.168.248.165', '10.37.130.4'
    }
    # 在Windows下SplitCap的分割IP为192-168-0-1, Linux下为192.168.0.1
    for ip in list(malicious_ips):
        malicious_ips.add(ip.replace('.', '-', 3))
    
    raw_pcaps = find_files(dataset_dir)
    flow_list, label_list = [], []
    tmp_output_dir = './data/tmp'

    real_packet_length = (header_length // 2 - 12) + (payload_length // 2)
    flow_length = real_packet_length * num_flow_packets
    for pcap_path in tqdm.tqdm(raw_pcaps, total=len(raw_pcaps), desc='Process ISCX-Bot-2014'):
        split_pcap(pcap_path, tmp_output_dir, num_flow_packets)
        
        pcap_files = find_files(tmp_output_dir)
        for pcap_file in tqdm.tqdm(pcap_files, total=len(pcap_files), desc=f'Process {pcap_path.rsplit("/", 1)[-1]}'):
            ips = process_ip(pcap_file, pcap_path)
            if ips is None:
                continue
            else:
                ip1, ip2 = ips

            try:   
                flow_data = process_pcap_to_array(pcap_file, num_flow_packets, header_length, payload_length)
            except Scapy_Exception:
                continue
            if flow_data is None: 
                continue
            
            label_list.append(1 if ip1 in malicious_ips or ip2 in malicious_ips else 0)
            flow_list.append(flow_data)
        shutil.rmtree(tmp_output_dir)
    flow_list = np.vstack(flow_list).reshape(len(flow_list), num_flow_packets, -1)
    label_list = np.asarray(label_list, dtype=np.int8)
    save_data(output_dir, flow_list, label_list)


def process_CIC_IOT_2022(dataset_dir: str, output_dir: str, num_flow_packets:int = 5, header_length: int = 160, payload_length: int = 104):
    label_names = ['Audio', 'Cameras', 'HomeAutomation', 'Flood', 'Hydra', 'Nmap']
    label_map = {label_name: label_index for label_index, label_name in enumerate(label_names)}

    def process_flows(pcap_root_dir, data_dir, flow_type, _flow_list, _label_list):
        matched_pcaps = find_files(pcap_root_dir)
        origin_num_samples = len(_flow_list)

        # Process such as Power-Audio
        for matched_pcap_path in tqdm.tqdm(matched_pcaps, total=len(matched_pcaps), desc=f'Process {data_dir.rsplit("-", 1)[1]}-{flow_type}'):
            split_pcap(matched_pcap_path, tmp_output_dir, num_flow_packets)
            pcap_files = find_files(tmp_output_dir)

            origin_pcap_file_name = matched_pcap_path.rsplit('/', 1)[-1]
            for pcap_file in pcap_files:
                if process_ip(pcap_file, origin_pcap_file_name) is None:
                    continue
                try:
                    flow_data = process_pcap_to_array(pcap_file, num_flow_packets, header_length, payload_length)
                except Scapy_Exception:
                    continue
                if flow_data is None:
                    continue
                _flow_list.append(flow_data)
            shutil.rmtree(tmp_output_dir)
        # 为处理过的该目录所有样本统一添加标签
        num_new_samples = len(_flow_list) - origin_num_samples
        _label_list.append(np.full(shape=(num_new_samples,), fill_value=label_map[flow_type], dtype=np.int8))
        print(f'Add {num_new_samples} in {data_dir.rsplit("-", 1)[1]}-{flow_type}')
        return _flow_list, _label_list

    flow_list, label_list = [], []
    tmp_output_dir = f'./data/tmp'

    for data_dir in ['1-Power', '3-Interactions']:
        for flow_type in ['Audio', 'Cameras', 'HomeAutomation']:
            dir_path = f'{dataset_dir}/{data_dir}/{flow_type}'
            flow_list, label_list = process_flows(dir_path, data_dir, flow_type, flow_list, label_list)

    data_dir = '6-Attacks'
    dir_path = f'{dataset_dir}/{data_dir}/1-Flood'
    flow_list, label_list = process_flows(dir_path, data_dir, 'Flood', flow_list, label_list)

    # Keep data_dir
    for flow_type in ['Hydra', 'Nmap']:
        dir_path = f'{dataset_dir}/{data_dir}/2-RTSPBruteForce/{flow_type}'
        flow_list, label_list = process_flows(dir_path, data_dir, flow_type, flow_list, label_list)

    flow_list = np.vstack(flow_list)
    label_list = np.concatenate(label_list)
    save_data(output_dir, flow_list, label_list)
    show_label_distribution(label_names, label_list)
    return flow_list, label_list


def show_label_distribution(label_names, label_list):
    total_samples = len(label_list)
    sample_counts = np.unique(label_list, return_counts=True)[1]
    print(sample_counts)
    for label_name, num_samples in zip(label_names, sample_counts):
        print(f'{label_name}: {num_samples}/{total_samples}({(num_samples / total_samples):.2%})')


def process_ToN_IoT(dataset_dir: str, output_dir: str, num_flow_packets:int = 5, header_length: int = 160, payload_length: int = 104):
    tmp_output_dir = './data/tmp'
    flow_list = []

    benign_data_dir = f'{dataset_dir}/normal_pcaps'
    benign_pcaps = find_files(benign_data_dir)
    flow_list = process_unlabeled_pcap(benign_pcaps, tmp_output_dir, num_flow_packets, header_length, payload_length,f'Process Benign', flow_list)
    # TODO other classes

    save_data(output_dir, flow_list, None)


def process_Bot_IoT(dataset_dir: str, output_dir: str, num_flow_packets:int = 5, header_length: int = 160, payload_length: int = 104):
    label_names = ['Benign', 'DoS', 'DDoS', 'Scan', 'Theft']
    label_map = {label_name: label_index for label_index, label_name in enumerate(label_names)}
    tmp_output_dir = './data/tmp'

    flow_list, label_list = [], []
    for class_name in tqdm.tqdm(label_names[1:], desc='Process Bot-IoT'):
        origin_num_samples = len(flow_list)
        flow_list = process_unlabeled_pcap(find_files(f'{dataset_dir}/{class_name}'), tmp_output_dir, num_flow_packets, header_length, payload_length,f'Process {class_name}', flow_list)
        label_list.append(np.full((len(flow_list)-origin_num_samples,), fill_value=label_map[class_name], dtype=np.int8))
    flow_list = np.vstack(flow_list)

    # benign_data = torch.load(f'{dataset_dir}/../ToN-IoT/data.pt')
    # flow_list = np.concatenate((flow_list, benign_data))
    # label_list.append(np.full((len(benign_data),), fill_value=label_map['Benign'], dtype=np.int8))
    label_list = np.concatenate(label_list)

    save_data(output_dir, flow_list, label_list)
    show_label_distribution(label_names[1:], label_list)
    print(f'Benign: {len(torch.load(f"{dataset_dir}/../ToN-IoT/data.pt"))}')


def process_unlabeled_pcap(pcaps, tmp_output_dir: str, num_flow_packets: int, header_length: int, payload_length: int, desc: str, flow_list):
    for matched_pcap_path in tqdm.tqdm(pcaps, total=len(pcaps), desc=desc):
        split_pcap(matched_pcap_path, tmp_output_dir, num_flow_packets)
        pcap_files = find_files(tmp_output_dir)

        origin_pcap_file_name = matched_pcap_path.rsplit('/', 1)[-1]
        for pcap_file in pcap_files:
            if process_ip(pcap_file, origin_pcap_file_name) is None:
                continue
            try:
                flow_data = process_pcap_to_array(pcap_file, num_flow_packets, header_length, payload_length)
            except Scapy_Exception:
                continue
            if flow_data is None:
                continue
            flow_list.append(flow_data)
        shutil.rmtree(tmp_output_dir)
    return flow_list


def process_pretrain_dataset(output_dir: str, num_flow_packets:int = 5, header_length: int = 160, payload_length: int = 104):
    flow_list = []
    tmp_output_dir = f'./data/tmp'

    real_packet_length = (header_length // 2 - 12) + (payload_length // 2)
    flow_length = real_packet_length * num_flow_packets
    for dataset in ['ISCX-2012', 'CIC-IDS-2017']:
        dataset_path = f'../flow_test/data/{dataset}/raw'
        matched_pcaps = find_files(dataset_path)
        flow_list = process_unlabeled_pcap(matched_pcaps, tmp_output_dir, num_flow_packets, header_length, payload_length, f'Process {dataset}', flow_list)

    flow_list = np.vstack(flow_list).reshape(len(flow_list), num_flow_packets, -1)
    print(f'Total {len(flow_list)} samples.')
    flow_list = torch.as_tensor(flow_list, dtype=torch.uint8)
    save_data(output_dir, flow_list)


def save_data(save_dir, flow_data, label=None):
    os.makedirs(save_dir, exist_ok=True)
    flow_data = torch.as_tensor(flow_data, dtype=torch.uint8)
    torch.save(flow_data, f'{save_dir}/data.pt')

    if label is not None:
        label = torch.as_tensor(label, dtype=torch.int8)
        torch.save(label, f'{save_dir}/label.pt')


def load_pretrain_data(num_flow_packets: int, packet_length: int, batch_size: int):
    data = torch.load('./data/pretrain/data.pt')[:, :num_flow_packets, :packet_length]
    data = data.to(torch.float32, non_blocking=True)
    from torchvision.transforms import functional as F
    data = F.normalize(data / 255., mean=[0.5], std=[0.5], inplace=True)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    return loader


def load_data(dataset_name: str, num_flow_packets: int, packet_length: int, batch_size: int, seed: int = 3407):
    dataset_dir = f'./data/{dataset_name}'
    data = torch.load(f'{dataset_dir}/data.pt').reshape(-1, num_flow_packets, packet_length)
    #[:, :num_flow_packets, :packet_length]
    label = torch.load(f'{dataset_dir}/label.pt').numpy()

    import torchvision.transforms.functional as F
    data = F.normalize(data.to(torch.float32) / 255., mean=[0.5], std=[0.5], inplace=True).numpy()
    num_classes = len(np.unique(label))

    from sklearnex.model_selection import train_test_split
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=seed,
                                                                      stratify=label)
    val_data, test_data, val_label, test_label = train_test_split(test_data, test_label, test_size=0.5,
                                                                  random_state=seed, stratify=test_label)

    train_loader = DataLoader(TensorDataset(torch.as_tensor(train_data), torch.as_tensor(train_label)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.as_tensor(val_data), torch.as_tensor(val_label)), batch_size=2048, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.as_tensor(test_data), torch.as_tensor(test_label)), batch_size=2048, shuffle=False)
    return train_loader, val_loader, test_loader, num_classes



if __name__ == '__main__':
    # input_file = 'data/test.pcap'
    # output_dir = 'data/test_split'
    num_flow_packets = 5
    raw_header_length_hex = 160
    raw_payload_length_hex = 104
    # dataset_dir = '../flow_test/data/ISCX-Bot-2014/raw'
    # output_dir = './data/ISCX-Bot-2014/tmp'
    # process_ISCX_Bot_dataset(dataset_dir, output_dir)

    # dataset_dir = '/data/yhy/project/CIC_IOT_2022'
    # output_dir = './data/CIC-IOT-2022'
    # process_CIC_IOT_2022(dataset_dir, output_dir, num_flow_packets, raw_header_length_hex, raw_payload_length_hex)

    # output_dir = './data/pretrain'
    # process_pretrain_dataset(output_dir, num_flow_packets, raw_header_length_hex, raw_payload_length_hex)

    input_dir = '/data3/lph/dataset/IOT/ToN_IoT'
    save_dir = './data/ToN_IoT'
    process_ToN_IoT(input_dir, save_dir, num_flow_packets, raw_header_length_hex, raw_payload_length_hex)

    input_dir = '/data3/lph/dataset/IOT/Bot-IoT'
    save_dir = './data/Bot_IoT'
    process_Bot_IoT(input_dir, save_dir, num_flow_packets, raw_header_length_hex, raw_payload_length_hex)
    
    # split_pcap(input_file, output_dir, num_flow_packets)
    
    # flow_list = []
    # pcap_files = find_files(output_dir)
    # for pcap_file in tqdm.tqdm(pcap_files):
    #     flow_data = process_pcap_to_array(pcap_file)
    #     if flow_data is None: 
    #         continue
    #     flow_list.append(flow_data)
    # flow_list = np.vstack(flow_list)
    # print(flow_list[0].reshape(5, -1))
    # print(flow_list.shape)
