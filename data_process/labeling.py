import os
import pandas as pd
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from filelock import FileLock
from collections import defaultdict
import pickle
import gc  # 垃圾回收模块
import time
import pyshark
from datetime import datetime,timedelta

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义输入和输出路径
input_csv = '/data/users/lph/datasets/IOT/ToN_IoT/SecurityEvents_Network_datasets/scanning8.csv'  # CSV 文件路径
input_pcap_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Split/normal_attack/password_normal/'  # 分割后的 PCAP 文件目录
output_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Label2'  # 输出目录
lock_file = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Label2/lockfile.lock'  # 锁文件路径
cache_file = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Label2/password_normal_cache.pkl'  # 缓存文件路径

# 创建一个字典来存储 pcap 文件的五元组信息，键为 (proto, src_ip, src_port, dst_ip, dst_port)，值为文件路径
pcap_index = defaultdict(list)


def get_flow_info_from_filename(filename):
    """从文件名中提取五元组信息（假设文件名格式为：MITM_normal1.pcap.TCP_5-150-255-234_43_192-168-1-180_49590.pcap）"""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('.')

    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")

    # 忽略前缀部分（如 MITM_normal1.pcap）
    flow_info_part = parts[-1]  # 获取五元组部分

    # 分割五元组信息
    proto, src_ip, src_port, dst_ip, dst_port = flow_info_part.split('_')

    # 将连字符分隔的 IP 地址转换为点分十进制格式
    src_ip = src_ip.replace('-', '.')
    dst_ip = dst_ip.replace('-', '.')

    # 将协议名称转为小写
    proto = proto.lower()

    return {
        'proto': proto,
        'src_ip': src_ip,
        'src_port': int(src_port),
        'dst_ip': dst_ip,
        'dst_port': int(dst_port)
    }


def load_cache(cache_file):
    """加载缓存文件，返回缓存的五元组信息和文件的最后修改时间"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            return cached_data['index'], cached_data['file_times']
    else:
        return defaultdict(list), {}


def save_cache(cache_file, index, file_times):
    """保存缓存文件，包含五元组信息和文件的最后修改时间"""
    with open(cache_file, 'wb') as f:
        pickle.dump({'index': index, 'file_times': file_times}, f)


def build_pcap_index(input_pcap_dir, cache_file):
    """构建 pcap 文件的索引，记录每个五元组对应的文件路径，并支持增量更新"""
    global pcap_index

    # 加载缓存
    pcap_index, file_times = load_cache(cache_file)

    # 获取当前目录下所有 pcap 文件的路径
    current_files = set()
    for root, dirs, files in os.walk(input_pcap_dir):
        for file in files:
            if file.endswith('.pcap'):
                pcap_path = os.path.join(root, file)
                current_files.add(pcap_path)

    # 检查是否有新的文件或修改过的文件
    for pcap_path in current_files:
        try:
            last_modified_time = os.path.getmtime(pcap_path)
            if pcap_path not in file_times or file_times[pcap_path] != last_modified_time:
                # 文件是新的或已修改，重新解析并更新索引
                flow_info = get_flow_info_from_filename(os.path.basename(pcap_path))
                key = (flow_info['proto'], flow_info['src_ip'], flow_info['src_port'], flow_info['dst_ip'],
                       flow_info['dst_port'])
                pcap_index[key].append(pcap_path)

                # 也记录反向五元组
                reverse_key = (flow_info['proto'], flow_info['dst_ip'], flow_info['dst_port'], flow_info['src_ip'],
                               flow_info['src_port'])
                pcap_index[reverse_key].append(pcap_path)

                # 更新文件的最后修改时间
                file_times[pcap_path] = last_modified_time
                logging.info(f"Updated index for {pcap_path}")
        except Exception as e:
            logging.error(f"Error processing file {pcap_path}: {e}")

    # 保存更新后的缓存
    save_cache(cache_file, pcap_index, file_times)


def get_flow_info_from_pcap(pcap_file_path):
    """从PCAP文件中提取最早的包的时间戳和五元组信息"""
    cap = pyshark.FileCapture(pcap_file_path)
    first_packet = None

    try:
        # 使用一个循环来找到第一个有效的数据包
        for packet in cap:
            if hasattr(packet, 'ip') and (hasattr(packet, 'tcp') or hasattr(packet, 'udp')):
                first_packet = packet
                break
    except Exception as e:
        logging.error(f"Error reading packets from {pcap_file_path}: {e}")
    finally:
        cap.close()  # 确保即使发生异常也会关闭 FileCapture 对象

    if not first_packet:
        raise ValueError(f"No valid packets found in {pcap_file_path}")

    # logging.debug(f"Packet info: {first_packet}")

    # 尝试获取时间戳，考虑不同的可能字段名
    timestamp = None
    try:
        # 注释掉的 frame_info 属性列表打印
        # logging.debug(f"Frame info attributes: {dir(first_packet.frame_info)}")

        if hasattr(first_packet, 'frame_info'):
            if hasattr(first_packet.frame_info, 'time_epoch'):
                timestamp = float(first_packet.frame_info.time_epoch)
            else:
                logging.error(
                    f"Could not find a suitable timestamp field in frame_info. Available fields: {dir(first_packet.frame_info)}")
                raise AttributeError("No suitable timestamp field found.")
        else:
            logging.error(f"Could not find frame_info in the packet. Packet structure: {dir(first_packet)}")
            raise AttributeError("No frame_info found in packet.")
    except AttributeError as e:
        logging.error(f"Error extracting timestamp from {pcap_file_path}: {e}")
        raise

    if timestamp is not None:
        timestamp = datetime.fromtimestamp(timestamp)  # 转换为datetime对象
    else:
        raise ValueError(f"Failed to extract timestamp from {pcap_file_path}")

    src_ip = first_packet.ip.src
    dst_ip = first_packet.ip.dst
    if hasattr(first_packet, 'tcp'):
        proto = 'tcp'
        src_port = int(first_packet.tcp.srcport)
        dst_port = int(first_packet.tcp.dstport)
    elif hasattr(first_packet, 'udp'):
        proto = 'udp'
        src_port = int(first_packet.udp.srcport)
        dst_port = int(first_packet.udp.dstport)
    else:
        raise ValueError(f"Unsupported protocol in the first packet of {pcap_file_path}")

    return {
        'proto': proto.lower(),
        'src_ip': src_ip,
        'src_port': src_port,
        'dst_ip': dst_ip,
        'dst_port': dst_port,
        'timestamp': timestamp
    }


def find_type_in_csv(flow_info, df, time_tolerance=timedelta(seconds=20)):
    """
    根据五元组信息在 CSV 文件中查找对应的 type，支持正向和反向匹配，并使用缓存。
    增加了基于时间戳的匹配，允许一定的时间误差（默认为 30 秒）。
    """
    unix_timestamp = int(flow_info['timestamp'].timestamp())  # 将 'timestamp' 转换为 Unix 时间戳

    # 将 DataFrame 中的协议字段转换为小写
    df['proto'] = df['proto'].str.lower()

    # 正向匹配
    mask_forward = (
            (df['src_ip'] == flow_info['src_ip']) &
            (df['src_port'] == flow_info['src_port']) &
            (df['dst_ip'] == flow_info['dst_ip']) &
            (df['dst_port'] == flow_info['dst_port']) &
            (df['proto'] == flow_info['proto']) &
            (abs(df['ts'].astype(int) - unix_timestamp) <= time_tolerance.total_seconds())
    )

    matching_rows_forward = df[mask_forward]
    if not matching_rows_forward.empty:
        return matching_rows_forward.iloc[0]['type']

    # 反向匹配
    mask_reverse = (
            (df['src_ip'] == flow_info['dst_ip']) &  # 源 IP 与目的 IP 对调
            (df['src_port'] == flow_info['dst_port']) &  # 源端口与目的端口对调
            (df['dst_ip'] == flow_info['src_ip']) &  # 目的 IP 与源 IP 对调
            (df['dst_port'] == flow_info['src_port']) &  # 目的端口与源端口对调
            (df['proto'] == flow_info['proto']) &  # 协议保持不变
            (abs(df['ts'].astype(int) - unix_timestamp) <= time_tolerance.total_seconds())  # 时间戳匹配
    )

    matching_rows_reverse = df[mask_reverse]
    if not matching_rows_reverse.empty:
        return matching_rows_reverse.iloc[0]['type']

    return None



def move_pcap_to_type_folder(pcap_file, output_dir, flow_type):
    """将 PCAP 文件移动到以 flow_type 为名的子文件夹中，使用文件锁确保线程安全"""
    pcap_path = Path(pcap_file)
    output_subdir = Path(output_dir) / flow_type

    # 使用文件锁确保创建子文件夹和移动文件的操作是线程安全的
    with FileLock(lock_file):
        # 创建子文件夹（如果不存在）
        os.makedirs(output_subdir, exist_ok=True)

        # 移动文件
        destination = output_subdir / pcap_path.name
        shutil.move(str(pcap_path), str(destination))
        # logging.info(f"Moved {pcap_path} to {destination}")


def process_single_pcap(pcap_file_path, df, output_dir):
    """处理单个 PCAP 文件"""
    try:
        flow_info = get_flow_info_from_pcap(pcap_file_path)
        logging.info(f"Processing file {pcap_file_path} with flow info: {flow_info}")
        flow_type = find_type_in_csv(flow_info, df)

        if flow_type:
            move_pcap_to_type_folder(pcap_file_path, output_dir, flow_type)
        else:
            logging.warning(f"No matching type found for {pcap_file_path}")
    except Exception as e:
        logging.error(f"Error processing file {pcap_file_path}: {e}")


def generate_pcap_file_paths(input_pcap_dir):
    """生成器函数，逐个生成 PCAP 文件路径"""
    for root, dirs, files in os.walk(input_pcap_dir):
        for file in files:
            if file.endswith('.pcap'):
                yield os.path.join(root, file)


def process_pcap_files_parallel(input_pcap_dir, output_dir, df, max_workers=None, batch_size=100):
    """使用多进程并行处理 PCAP 文件，采用流式处理方式，并使用文件锁和缓存"""
    # 构建或更新 pcap 文件索引
    build_pcap_index(input_pcap_dir, cache_file)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 使用 ProcessPoolExecutor 并行处理文件
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        batch_count = 0

        # 逐个生成文件路径，并分批提交任务
        for pcap_file_path in generate_pcap_file_paths(input_pcap_dir):
            future = executor.submit(process_single_pcap, pcap_file_path, df, output_dir)
            futures.append(future)
            batch_count += 1

            # 每处理 batch_size 个文件后，等待当前批次完成
            if batch_count >= batch_size:
                for future in as_completed(futures):
                    try:
                        result = future.result()  # 获取任务的返回值
                        if result is not None:
                            logging.info(f"Task completed with result: {result}")
                        else:
                            logging.debug(f"Task completed with no result (None).")
                    except Exception as e:
                        logging.error(f"An error occurred during processing: {e}")

                # 清理已完成的任务
                del futures[:]
                batch_count = 0

                # 手动触发垃圾回收
                gc.collect()

        # 处理剩余的任务
        if futures:
            for future in as_completed(futures):
                try:
                    result = future.result()  # 获取任务的返回值
                    if result is not None:
                        logging.info(f"Task completed with result: {result}")
                    else:
                        logging.debug(f"Task completed with no result (None).")
                except Exception as e:
                    logging.error(f"An error occurred during processing: {e}")

            # 清理已完成的任务
            del futures[:]

            # 手动触发垃圾回收
            gc.collect()


if __name__ == '__main__':
    # 设置进程池大小（可以根据 CPU 核心数调整）
    max_workers = 64  # 通常是 CPU 核心数的 1-2 倍

    # 设置批量处理的大小
    batch_size = 16000  # 每次处理 100 个文件

    # 读取 CSV 文件
    df = pd.read_csv(input_csv)

    # # 打印 CSV 文件的前几行以检查数据格式
    # logging.info("First few rows of the CSV file:")
    # logging.info(df.head())

    # 处理 PCAP 文件
    process_pcap_files_parallel(input_pcap_dir, output_dir, df, max_workers, batch_size)