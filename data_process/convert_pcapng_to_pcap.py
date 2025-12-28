# import os
# import subprocess
# import logging
# from pathlib import Path
#
# # 设置日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# def convert_pcapng_to_pcap(input_dir, output_dir):
#     """
#     递归遍历输入目录中的所有 .pcapng 文件，并使用 tshark 将其转换为 .pcap 文件。
#     保持原始的目录结构和文件名，仅更改文件后缀。
#
#     参数:
#     input_dir (str): 输入目录路径，包含 .pcapng 文件。
#     output_dir (str): 输出目录路径，保存转换后的 .pcap 文件。
#     """
#     # 确保输出目录存在
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#     # 递归遍历输入目录
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if (file.endswith('.pcapng') or file.endswith('.pcap')):
#                 # 构建输入文件的完整路径
#                 input_file_path = os.path.join(root, file)
#
#                 # 计算相对路径，以保持目录结构
#                 relative_path = os.path.relpath(root, input_dir)
#                 output_subdir = os.path.join(output_dir, relative_path)
#
#                 # 确保输出子目录存在
#                 Path(output_subdir).mkdir(parents=True, exist_ok=True)
#
#                 # 构建输出文件的完整路径，更改后缀为 .pcap
#                 output_file_path = os.path.join(output_subdir, f"{file[:-7]}.pcap")
#
#                 # 调用 tshark 进行转换
#                 try:
#                     command = ['tshark', '-r', input_file_path, '-F', 'pcap', '-w', output_file_path]
#                     subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                     logging.info(f"Converted {input_file_path} to {output_file_path}")
#                 except subprocess.CalledProcessError as e:
#                     logging.error(f"Failed to convert {input_file_path}: {e.stderr.decode().strip()}")
#
# if __name__ == "__main__":
#     # 示例参数
#     input_directory = '/data3/lph/dataset/IOT/ToN_IoT'  # 输入目录路径，包含 .pcapng 文件
#     output_directory = '/data3/lph/dataset/IOT/ToN_IoT_NoPcapng'  # 输出目录路径，保存转换后的 .pcap 文件
#
#     # 调用函数进行批量转换
#     convert_pcapng_to_pcap(input_directory, output_directory)

import os
import subprocess
import logging
from pathlib import Path

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_file_integrity(input_file_path):
    """
    使用 capinfos 检查文件完整性，返回 True 如果文件完整，False 如果文件被截断。
    """
    try:
        result = subprocess.run(['capinfos', '-E', input_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "File appears to be cut short" in result.stderr:
            logging.warning(f"File {input_file_path} appears to be cut short.")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check integrity of {input_file_path}: {e.stderr.strip()}")
        return False


def transform_pcapng_to_pcap(file_paths):
    input_file_path, output_file_path = file_paths
    # 调用 tshark 进行转换，即使文件不完整也继续转换
    try:
        command = ['tshark', '-r', input_file_path, '-F', 'pcap', '-w', output_file_path, '-Y', '!ipv6']
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # command = ['tshark', '-r', input_file_path, '-F', 'pcap', '-w', output_file_path, '-Y', '!ipv6']
        # result = subprocess.popen(command)
        # if "cut short" in result:
        #     logging.warning(f"File {input_file_path} appears to be cut short.")
        #     command2 = ['pcapfix', {input_file_path}]
        #     subprocess.run(command2, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #     result = subprocess.popen(command)

        # result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            logging.info(f"Successfully converted {input_file_path} to {output_file_path}")
        else:
            logging.warning(f"Conversion of {input_file_path} completed with warnings/errors: {result.stderr.strip()}")
    except Exception as e:
        logging.error(f"Failed to convert {input_file_path}: {e}")


def convert_pcapng_to_pcap(input_dir, output_dir):
    """
    递归遍历输入目录中的所有 .pcapng 文件，并使用 tshark 将其转换为 .pcap 文件。
    保持原始的目录结构和文件名，仅更改文件后缀。

    参数:
    input_dir (str): 输入目录路径，包含 .pcapng 文件。
    output_dir (str): 输出目录路径，保存转换后的 .pcap 文件。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    path_list = []
    # 递归遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pcapng') or file.endswith('.pcap'):
                # 构建输入文件的完整路径
                input_file_path = os.path.join(root, file)

                # 计算相对路径，以保持目录结构
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                # 确保输出子目录存在
                os.makedirs(output_subdir, exist_ok=True)

                # 构建输出文件的完整路径，更改后缀为 .pcap
                output_file_path = os.path.join(output_subdir, f"{file.rsplit('.', 1)[0]}.pcap")

                # # 检查文件完整性
                # is_complete = check_file_integrity(input_file_path)
                path_list.append((input_file_path, output_file_path))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(transform_pcapng_to_pcap, path_list)


if __name__ == "__main__":
    # 示例参数
    # input_directory = '/data3/lph/dataset/IOT/Bot_IoT'  # 输入目录路径，包含 .pcapng 文件
    # output_directory = '/data3/lph/dataset/IOT/Bot_IoT_Processed'  # 输出目录路径，保存转换后的 .pcap 文件
    input_directory = '/data/users/lph/projects/IIOT_Incremental_Learning/data/CIC_IIoT_2025_raw3/'  # 输入目录路径，包含 .pcapng 文件
    output_directory = '/data/users/lph/projects/IIOT_Incremental_Learning/data/CIC_IIoT_2025_raw4/'  # 输出目录路径，保存转换后的 .pcap 文件


    # 调用函数进行批量转换
    convert_pcapng_to_pcap(input_directory, output_directory)