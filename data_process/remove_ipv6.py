import os
import subprocess
from tqdm import tqdm

def find_files(directory, extension='.pcap'):
    """递归查找指定目录下的所有PCAP文件"""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def filter_ipv6(pcap_file, output_file):
    """使用 tshark 过滤掉 IPv6 数据包"""
    try:
        # 使用 tshark 过滤掉 IPv6 数据包
        command = ['tshark', '-r', pcap_file, '-Y', '!ipv6', '-w', output_file]
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Filtered {pcap_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {pcap_file}: {e.stderr.decode().strip()}")

def process_pcaps(input_dir, output_base_dir):
    """处理一批PCAP文件，过滤掉IPv6数据包并保持原始目录结构"""
    pcaps = find_files(input_dir)

    for pcap_file in tqdm(pcaps, desc="Processing PCAP files"):
        # 获取原始PCAP文件的相对路径
        relative_path = os.path.relpath(pcap_file, start=input_dir)
        output_dir = os.path.join(output_base_dir, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)  # 创建对应的输出目录

        # 构建输出文件路径
        output_file = os.path.join(output_dir, os.path.basename(pcap_file))

        # 使用 tshark 过滤掉 IPv6 数据包
        filter_ipv6(pcap_file, output_file)

# 示例调用
if __name__ == '__main__':
    # input_dir = '/data3/lph/dataset/IOT/Bot_IoT_NoPcapng'  # 原始PCAP文件的根目录
    # output_base_dir = '/data3/lph/dataset/IOT/Bot_IoT_remove_ipv6'  # 保存过滤后PCAP文件的根目录
    input_dir = '/data3/lph/dataset/IOT/ToN_IoT_NoPcapng'  # 原始PCAP文件的根目录
    output_base_dir = '/data3/lph/dataset/IOT/ToN_IoT_remove_ipv6'  # 保存过滤后PCAP文件的根目录

    process_pcaps(input_dir, output_base_dir)