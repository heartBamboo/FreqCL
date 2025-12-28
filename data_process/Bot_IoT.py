
import os
import numpy as np
import tqdm
import torch
import shutil
import scapy
from scapy.all import rdpcap
from scapy.error import Scapy_Exception
import subprocess
import binascii

def find_files(data_path, extension=".pcap", max_files_total=30000):
    """
    查找当前目录下的所有指定扩展名的文件，最多取 max_files_total 个文件。

    Args:
        data_path (str): 当前目录路径。
        extension (str, optional): 文件扩展名，默认为 ".pcap"。
        max_files_total (int, optional): 当前目录最多取的文件数，默认为 10000。

    Returns:
        list: 所有符合条件的文件路径列表。
    """
    pcap_files = []
    
    # 列出当前目录下的所有文件
    try:
        files = os.listdir(data_path)
    except FileNotFoundError:
        print(f"Directory not found: {data_path}")
        return pcap_files
    
    # 初始化计数器
    count = 0
    
    for file in files:
        # 如果文件扩展名匹配且未达到最大数量限制
        if file.endswith(extension):
            pcap_files.append(os.path.join(data_path, file))
            count += 1
            
            # 如果达到了 max_files_total，则停止遍历
            if count >= max_files_total:
                break
    
    return pcap_files


ignored_ips = {'0.0.0.0', '255.255.255.255'}
def ignored_ip(ips):
    return any(ip in ignored_ips or ip.startswith('224') for ip in ips)
    # return any(ip in ignored_ips for ip in ips)

# def process_ip(pcap_file_path, origin_pcap_file_name):
#     pcap_file_name = pcap_file_path.rsplit('/', 1)[-1]

#     if ':' in pcap_file_name:
#         return None
#     # test.pcap.TCP_13-35-210-49_443_172-26-248-34_63662.pcap
#     _, ip1, _, ip2, _ = pcap_file_name.split(origin_pcap_file_name, 1)[-1].split('_')
#     if ignored_ip([ip1, ip2]):
#         return None
#     return ip1, ip2

def process_ip(pcap_file_path):
    try:
        parts = pcap_file_path.split('.', 1)
        origin_pcap_file_name = parts[0].rsplit('/', 1)[-1] + '.pcap'

        # 提取文件名（去掉路径）
        pcap_file_name = pcap_file_path.rsplit('/', 1)[-1]

        # 如果文件名中包含冒号，直接返回 None
        if ':' in pcap_file_name:
            return None

        # 去掉前面的 origin_pcap_file_name 部分，保留剩余部分
        remaining_part = pcap_file_name.split(origin_pcap_file_name, 1)[-1]

        # 按照 '_' 分割
        split_result = remaining_part.split('_')

        # 打印调试信息
        # print("origin_pcap_file_name:", origin_pcap_file_name)
        # print("Split result:", split_result)

        # 确保分割结果有 5 个部分
        if len(split_result) == 5:
            ip1 = split_result[1]  # 第二部分是第一个 IP 地址
            ip2 = split_result[3]  # 第四部分是第二个 IP 地址

            # 检查 IP 是否在忽略列表中
            if ignored_ip([ip1, ip2]):
                return None

            return ip1, ip2
        else:
            raise ValueError(f"Unexpected file name format: {pcap_file_name}")

    except Exception as e:
        print(f"Error processing file {pcap_file_path}: {e}")
        return None

def string_to_array(flow_string):
    return np.asarray([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)], dtype=np.uint8)

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
    #real_header_length = header_length // 2
    # 算数据包的长度，单位转换为字节
    real_packet_length = real_header_length + payload_length // 2

    # 将解析得到的数据包头部转换成字节数组，并去掉IP头和端口号
    header = string_to_array(header[:header_length])
    header = np.concatenate((header[:12], header[24:]))

    # 如果没有负载，则填充0
    if payload is None:
        packet_data = np.pad(header, (0, real_packet_length - len(header)))
    else:
        payload = string_to_array(payload[:payload_length])
        # 如果长度无需填充则直接拼接返回
        if len(header) == real_header_length and len(payload) == payload_length // 2:
            packet_data = np.concatenate((header, payload))
        else:
            # 隐式填充: 构造全0数组再将有效值赋值到对应位置, 相比pad减少了临时数组数量
            packet_data = np.zeros(real_packet_length, dtype=np.uint8)
            packet_data[:len(header)] = header
            packet_data[real_header_length: real_header_length + len(payload)] = payload
    return packet_data


def process_pcap_to_array(pcap_filename, num_flow_packets=5, header_length=160, payload_length=104):
    """将pcap文件转换为数组

    Args:
        pcap_filename (_type_): _description_
        num_flow_packets (int, optional): _description_. Defaults to 5.

    Returns: 网络流字节数组
    """
    real_packet_length = (header_length // 2 - 12) + (payload_length // 2)
    # real_packet_length = (header_length + payload_length) // 2
    flow_length = real_packet_length * num_flow_packets
    first_ip = None
    packet_directions = []
    five_tuple = None

    # 解析pcap文件，保留前num_flow_packets个包
    #packets = scapy.rdpcap(pcap_filename)[:num_flow_packets]
    try:
        # 验证文件是否存在
        if not os.path.exists(pcap_filename):
            print(f"File not found: {pcap_filename}")
            return None, None, None

        # 使用 rdpcap 读取 pcap 文件
        packets = rdpcap(pcap_filename)[:num_flow_packets]
        if len(packets) == 0:
            print(f"Error processing file {pcap_filename}: No data could be read!")
            return None, None, None

    except Exception as e:
        # 捕获所有异常，记录错误信息并跳过损坏的文件
        print(f"Error processing file {pcap_filename}: {e}")
        return None, None, None

    first_ip = None
    packet_directions = []
    five_tuple = None
    flow_data = []

    for packet_index, packet in enumerate(packets):
        # 遍历packets，将每个packet转成字节数组
        try:
            packet_data = packet_to_array(packet, header_length, payload_length)

            if packet_index == 0:
                ip_layer = packet["IP"]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst

                if "TCP" in packet:
                    tcp_layer = packet["TCP"]
                    src_port = tcp_layer.sport
                    dst_port = tcp_layer.dport
                    five_tuple = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_TCP"

                elif "UDP" in packet:
                    udp_layer = packet["UDP"]
                    src_port = udp_layer.sport
                    dst_port = udp_layer.dport
                    five_tuple = f"{src_ip}_{src_port}_{dst_ip}_{dst_port}_UDP"
                else:
                    five_tuple = f"{src_ip}_0_{dst_ip}_0_Other"
                first_ip = src_ip
                packet_direction = 1
            else:
                packet_direction = 1 if packet["IP"].src == first_ip else 2
            packet_directions.append(packet_direction)

        except:
            # 解析过程中，如果第一个包解析失败，舍弃整个网络流。
            if packet_index == 0:
                return None, None, None
            # 解析过程中，如果后续的包解析失败，使用0填充该包对应的数据。
            packet_data = np.zeros(flow_length // num_flow_packets, dtype=np.uint8)
            packet_directions.append(0)
        flow_data.append(packet_data)
    # 将最多num_flow_packets个包拼接成对应的网络流数据
    flow_data = np.concatenate(flow_data)
    packet_directions = np.asarray(packet_directions, dtype=np.int8)
    # 如果网络流数据的长度不足flow_length，使用0填充。
    if len(flow_data) < flow_length:
        flow_data = np.pad(flow_data, (0, flow_length - len(flow_data)))
        packet_directions = np.pad(packet_directions, (0, num_flow_packets - len(packet_directions)))
    return flow_data, packet_directions, five_tuple


def process_unlabeled_pcap(pcaps, output_dir: str,  num_flow_packets: int, header_length: int,
                           payload_length: int, desc: str,
                           batch_size: int):
    for batch_index, start_index in enumerate(range(0, len(pcaps), batch_size)):
        batch_pcaps = pcaps[start_index:start_index + batch_size]
        batch_flow_list = []
        directions_list = []
        five_tuple_list = []

        #for matched_pcap_path in tqdm.tqdm(batch_pcaps, total=len(batch_pcaps), desc=desc):
            # split_pcap(matched_pcap_path, tmp_output_dir, num_flow_packets)
            # pcap_files = find_files(tmp_output_dir)
            # 现在已经分割好了，所以可以直接拿来进行下一步
        pcap_files = batch_pcaps



        for pcap_file in pcap_files:
            if process_ip(pcap_file) is None:
                continue
            try:
                flow_data, directions, five_tuple = process_pcap_to_array(pcap_file, num_flow_packets,
                                                                            header_length, payload_length)
            except Scapy_Exception:
                continue
            if flow_data is not None:
                batch_flow_list.append(flow_data)
                if (len(directions) != 8):
                    print('len')
                directions_list.append(directions)
                five_tuple_list.append(five_tuple)

            #shutil.rmtree(tmp_output_dir, ignore_errors=True)  # 删除临时文件夹

        # Save the batch of flows to disk and clear the list
        if len(batch_flow_list) > 0:
            batch_flow_list = np.vstack(batch_flow_list)
            directions_list = np.vstack(directions_list)
            save_batch_to_disk(batch_flow_list, output_dir, batch_index, directions_list, five_tuple_list)

def save_batch_to_disk(batch_flow_array, output_dir, batch_index, directions_list, five_tuple_list):
    os.makedirs(output_dir, exist_ok=True)
    batch_flow_tensor = torch.as_tensor(batch_flow_array, dtype=torch.uint8)
    directions_list = torch.as_tensor(directions_list, dtype=torch.int8)
    # five_tuple_list = torch.as_tensor(five_tuple_list, dtype=torch.int8)

    torch.save(batch_flow_tensor, f'{output_dir}/flow_{batch_index}.pt')
    torch.save(directions_list, f'{output_dir}/direction_{batch_index}.pt')
    torch.save(five_tuple_list, f'{output_dir}/five_tuple_{batch_index}.pt')


def process_ToN_IoT(dataset_dir: str, output_dir: str, num_flow_packets: int = 5, header_length: int = 160,
                    payload_length: int = 104, batch_size: int = 100, five_tuple_output_dir: str = None):


    #root_data_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Select'
    #save_root_data_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT'
    label_names = os.listdir(dataset_dir)
    label_names = ['DDoS_UDP',  'DoS_HTTP', 'DoS_TCP', 'DDoS_HTTP', 'Keylogging', 'Scan_Service', 'Scan_OS', 'Data_Exfiltration','DDoS_TCP', 'DoS_UDP']

    # for i in range(len(label_names)):
    #     data_dir = f'{dataset_dir}/{label_names[i]}'
    #     save_dir = f'{output_dir}/{label_names[i]}'
    #     pcaps = find_files(data_dir,".pcap", batch_size)
    #     process_unlabeled_pcap(pcaps, save_dir, num_flow_packets, header_length,
    #                                    payload_length, f'Process Benign',
    #                                    batch_size)
    # TODO other classes

    # label_map = {'benign': 0}
    label_map = {'DDoS_UDP': 0} # bot-iot 没有benign
    for label_name in label_names:
        if label_name not in label_map:
            label_map[label_name] = len(label_map)
    reverse_label_map = {v: k for k, v in label_map.items()}

    full_data, full_label = [], []
    for label_name in label_names:
        data = torch.load(f'{output_dir}/{label_name}/flow_0.pt')
        label = torch.full((len(data),), fill_value=label_map[label_name], dtype=torch.int8)
        full_data.append(data)
        full_label.append(label)
    full_data = torch.cat(full_data)
    full_label = torch.cat(full_label)
    label_map = reverse_label_map

    full_data_np = full_data.cpu().numpy()
    full_label_np = full_label.cpu().numpy()

    # 保存为 NumPy 格式
    np.save(f'{output_dir}/data.npy', full_data_np)
    np.save(f'{output_dir}/label.npy', full_label_np)
    torch.save(label_map, f'{output_dir}/label_map.pt')

    #save_data(output_dir, flow_list, None)


if __name__ == '__main__':
    num_flow_packets = 8
    raw_header_length_hex = 88  # 这里是（32+12） * 2 （32+12是加上了ip和端口的长度，里面会处理掉ip和端口，但是这里得先加上）
    raw_payload_length_hex = 192 # 这里是96 * 2 （不知道为什么要乘2才能达到原本的设定）
    batch_size = 20000

    input_dir = '/data/users/lph/datasets/IOT/Bot_IoT/Bot_IoT_Split'
    save_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/Bot-IoT_20000'
    five_tuple_output_dir =  '/data/users/lph/projects/IIOT_Incremental_Learning/data/Bot-IoT_20000'
    process_ToN_IoT(input_dir, save_dir, num_flow_packets, raw_header_length_hex, raw_payload_length_hex, batch_size, five_tuple_output_dir)

