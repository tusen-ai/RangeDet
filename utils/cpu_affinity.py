import psutil
import os
import subprocess
import logging


def simple_bind_cpus(rank, num_partition, logical=False):
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu_count = psutil.cpu_count(logical=logical)
    cpu_count_per_worker = cpu_count // num_partition
    cpu_list = list(range(rank * cpu_count_per_worker, (rank + 1) * cpu_count_per_worker))
    print("bind cpu list:{}".format(cpu_list))
    p.cpu_affinity(cpu_list)
    logging.info("rank: {}, pid:{}, affinity to cpu {}".format(rank, pid, cpu_list))


def simple_bind_cpus_with_superthread(rank, num_partition):
    pid = os.getpid()
    p = psutil.Process(pid)
    phy_cpu_count = psutil.cpu_count(logical=False)
    cpu_count_per_worker = phy_cpu_count // num_partition
    cpu_list = list(range(rank * cpu_count_per_worker, (rank + 1) * cpu_count_per_worker))
    cpu_list += list(
        range(phy_cpu_count + rank * cpu_count_per_worker, phy_cpu_count + (rank + 1) * cpu_count_per_worker))
    p.cpu_affinity(cpu_list)
    logging.info("rank: {}, pid:{}, affinity to cpu {}".format(rank, pid, cpu_list))


def bind_cpus_with_list(cpu_list):
    pid = os.getpid()
    p = psutil.Process(pid)
    p.cpu_affinity(cpu_list)
    logging.info("pid:{}, affinity to cpu {}".format(pid, cpu_list))


def bind_cpus_on_ecos(rank, num_partition):
    pid = os.getpid()
    p = psutil.Process(pid)
    allowed_list = cpu_allowed_list()
    if rank == 0:
        print("cpu allowed list len:{}, {}".format(len(allowed_list), allowed_list))
    cpu_count_per_worker = len(allowed_list) // num_partition
    cpu_list = allowed_list[int(rank * cpu_count_per_worker):int((rank + 1) * cpu_count_per_worker)]
    p.cpu_affinity(cpu_list)
    logging.info("rank: {}, pid:{}, affinity to cpu {}".format(rank, pid, cpu_list))


def cpu_allowed_list():
    byte_info = subprocess.check_output("cat /proc/$$/status|grep Cpus_allowed_list|awk '{print $2}'", shell=True)
    cpu_list = byte_info.decode("utf-8").replace("\n", "").split(",")
    allowed_list = []
    for item in cpu_list:
        ranges = [int(cpuid) for cpuid in item.split('-')]
        if len(ranges) == 1:
            allowed_list.append(ranges[0])
        else:
            allowed_list += list(range(ranges[0], ranges[1] + 1))
    return allowed_list
