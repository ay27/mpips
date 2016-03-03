import sys
from random import randint

from mpi4py import MPI

from mpips import load


class MpiPs(object):
    """
    A parameter server implementation based on PY4MPI.
    """

    def __init__(self, ps_num):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.ps_num = ps_num
        self.group = self.comm.group
        # make a group for ps
        self.ps_group = self.group.Range_incl([(0, ps_num-1, 1), ])
        self.ps_comm = self.comm.Create(self.ps_group)
        self.ps_group.Free()
        # make a group for worker
        self.wk_group = self.group.Range_excl([(0, ps_num-1, 1), ])
        self.wk_comm = self.comm.Create(self.wk_group)
        self.wk_group.Free()

        self.is_ps = (self.rank < ps_num)
        if self.is_ps:
            self._kv_store = {}
            self._run_ps()
        else:
            self.wk_num = self.wk_comm.Get_size()
            self.wk_rank = self.wk_comm.Get_rank()

    def sync(self):
        self.wk_comm.barrier()

    def end(self):
        self.sync()
        if self.wk_rank == 0:
            for i in range(self.ps_num):
                self.comm.send("", dest=i, tag=1)
        self.sync()
        sys.exit(0)

    def _get_dest_from_key(self, key):
        return hash(key) % self.ps_num

    def _gen_tag(self):
        return randint(1, 102400)

    def push_vector(self, key, vec):
        data = {
            "key": key,
            "vec": vec,
        }
        dest = self._get_dest_from_key(key)
        self.comm.send(data, dest=dest, tag=1)
        return True

    def pull_vector(self, key):
        dest = self._get_dest_from_key(key)
        tag = self._gen_tag()
        self.comm.send(key, dest=dest, tag=tag)
        return self.comm.recv(source=dest, tag=tag)

    def _run_ps(self):
        while True:
            st = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=st)
            if isinstance(data, dict):
                key = data.get("key")
                vec = data.get("vec")
                if key not in self._kv_store:
                    self._kv_store[key] = vec
                else:
                    self._kv_store[key] += vec
            elif data:
                vec = self._kv_store.get(data, None)
                self.comm.send(vec, dest=st.source, tag=st.tag)
            else:
                sys.exit(0)

    def load_input(self, input_file):
        return load.load(input_file, self.wk_rank, self.wk_num)
