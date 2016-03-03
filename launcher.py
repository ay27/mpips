import os
import time
import subprocess

TASK_NUM = 10
TASK_FILE = 'alg/lr.py'
CMD_RUN_MPI_TASK = 'mpirun -np %d python3.5 %s' % (TASK_NUM, TASK_FILE)

if __name__ == '__main__':
    p = subprocess.Popen(CMD_RUN_MPI_TASK, shell=True)
    p.communicate()
    # os.popen(CMD_RUN_MPI_TASK)
