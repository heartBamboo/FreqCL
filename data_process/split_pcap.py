import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import gc  # 引入垃圾回收模块
import psutil  # 用于监控内存使用
import time

BATCH_SIZE = 20  # 每批处理的文件数量
MIN_AVAILABLE_MEMORY = 1024 * 1024 * 1024  # 最小可用内存（1GB）
MAX_TOTAL_MEMORY = 40 * 1024 * 1024 * 1024  # 最大允许的总内存使用（40GB）


def split_pcap(input_file: str, output_dir: str, num_flow_packets: int):
    """按照五元组将pcap文件分割成网络流"""
    os.makedirs(output_dir, exist_ok=True)

    # 设置内存限制（例如 2GB）
    set_memory_limit(2 * 1024 * 1024)

    command = f'mono /data/users/lph/tools/SplitCap.exe -r {input_file} -o {output_dir} -s session -p {num_flow_packets}'
    try:
        subprocess.run(command, shell=True, check=True, close_fds=True)
        print(f"Successfully processed file: {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {input_file}: {e}")


def get_available_memory():
    """获取当前可用的物理内存（字节）"""
    mem = psutil.virtual_memory()
    return mem.available


def get_memory_usage(pid):
    """获取指定进程的内存使用量（RSS，字节）"""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss
    except psutil.NoSuchProcess:
        return 0


def monitor_memory_usage(futures, max_total_memory=MAX_TOTAL_MEMORY):
    """监控所有子进程的总内存使用量，确保不超过最大内存限制"""
    total_memory = 0
    for future in futures:
        if future.done():
            continue
        pid = future.pid  # 假设你已经保存了每个子进程的 PID
        memory_usage = get_memory_usage(pid)
        total_memory += memory_usage

        if total_memory > max_total_memory:
            print("Memory limit exceeded. Pausing processing...")
            return False

    return True


def process_batch(tasks, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(split_pcap, *task) for task in tasks]

        while not all(future.done() for future in futures):
            if not monitor_memory_usage(futures):
                time.sleep(5)  # 等待一段时间，让系统有时间释放内存
            else:
                time.sleep(1)  # 检查频率

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    # 批次处理结束后进行垃圾回收
    gc.collect()

    # 等待一段时间，让系统有时间释放缓存
    time.sleep(5)  # 根据实际情况调整等待时间


def process_pcap_files(input_dir: str, output_dir: str, num_flow_packets: int, initial_max_workers: int = 4):
    tasks = []
    available_memory = get_available_memory()
    max_workers = initial_max_workers

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pcap'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                tasks.append((input_file_path, output_subdir, num_flow_packets))

                if len(tasks) >= BATCH_SIZE:
                    process_batch(tasks, max_workers)
                    tasks = []

                    # 动态调整 max_workers
                    available_memory = get_available_memory()
                    max_workers = adjust_max_workers(available_memory, tasks)

                    # 确保有足够的可用内存继续处理
                    while available_memory < MIN_AVAILABLE_MEMORY:
                        print("Not enough available memory. Waiting for cache to be released...")
                        time.sleep(5)
                        available_memory = get_available_memory()

    if tasks:
        process_batch(tasks, max_workers)


if __name__ == '__main__':
    input_dir = '/data/users/lph/datasets/IOT/ToN_IoT_Processed'
    output_dir = '/data/users/lph/projects/IIOT_Incremental_Learning/data/ToN_IoT_Split2'
    num_flow_packets = 1000
    initial_max_workers = 4  # 初始最大进程数

    process_pcap_files(input_dir, output_dir, num_flow_packets, initial_max_workers)