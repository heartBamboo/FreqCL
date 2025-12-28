import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import gc  # 引入垃圾回收模块

BATCH_SIZE = 20  # 每批处理的文件数量

tmp_dir = "/data/users/lph/tmp"
os.makedirs(tmp_dir, exist_ok=True)

# 设置环境变量（会影响当前进程及其子进程）
os.environ["TMPDIR"] = tmp_dir


def split_pcap(input_file: str, output_dir: str, File_handle: int):
    """按照五元组将pcap文件分割成网络流

    Args:
        input_file (str): pcap文件路径
        output_dir (str): 分割后的网络流存放路径
        File_handle (int): 文件句柄数
    """
    os.makedirs(output_dir, exist_ok=True)

    command = f'mono /data/users/lph/tools/SplitCap.exe -r {input_file} -o {output_dir} -s session -p {File_handle}'
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully processed file: {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {input_file}: {e}")


def process_pcap_files(input_dir: str, output_dir: str, File_handle: int, max_workers: int = 4):
    """递归处理输入目录中的所有 .pcap 文件，并保持输出目录的相同结构关系

    Args:
        input_dir (str): 输入目录路径，包含 .pcap 文件
        output_dir (str): 输出目录路径，保存分割后的网络流
        File_handle (int): 网络流中的最大包数量
        max_workers (int): 最大进程数
    """
    tasks = []

    # 遍历输入目录及其所有子目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pcap'):
                # 构建输入文件的完整路径
                input_file_path = os.path.join(root, file)

                # 计算相对路径，以保持目录结构
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                # 确保输出子目录存在
                os.makedirs(output_subdir, exist_ok=True)

                # 将任务添加到列表中
                tasks.append((input_file_path, output_subdir, File_handle))

                # 当任务列表达到批次大小时，开始处理这批任务
                if len(tasks) >= BATCH_SIZE:
                    process_batch(tasks, max_workers)
                    tasks = []  # 清空任务列表

    # 处理剩余的任务
    if tasks:
        process_batch(tasks, max_workers)

def process_batch(tasks, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(split_pcap, *task) for task in tasks]

        # 等待所有任务完成
        for future in as_completed(futures):
            try:
                future.result()  # 获取结果，如果有异常会在此处抛出
            except Exception as e:
                print(f"An error occurred: {e}")

    # 批次处理结束后进行垃圾回收
    gc.collect()

if __name__ == '__main__':
    input_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/Edge_IIoT_raw/'
    output_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/Edge_IIoT_Split/'

    File_handle = 100
    max_workers = 32  # 可以根据你的系统配置调整进程数

    process_pcap_files(input_dir, output_dir, File_handle, max_workers)
