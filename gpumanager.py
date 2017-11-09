#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:15:33 2017

Part of code copied from gpustat

@author: eyulush
"""

import platform
import numpy as np
import os

from subprocess import check_output
from datetime import datetime

__version__ = '0.0.1.dev'

def execute_process(command_shell):
    stdout = check_output(command_shell, shell=True).strip()
    if not isinstance(stdout, (str)):
        stdout = stdout.decode()
    return stdout

class GPUStat(object):
    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError('entry should be a dict, {} given'.format(type(entry)))
        self.entry = entry
        self.processes = []

        for k in self.entry.keys():
            if 'Not Supported' in self.entry[k]:
                self.entry[k] = None

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    @property
    def uuid(self):
        """
        Returns the uuid returned by nvidia-smi,
        e.g. GPU-12345678-abcd-abcd-uuid-123456abcdef
        """
        return self.entry['uuid']

    @property
    def name(self):
        """
        Returns the name of GPU card (e.g. Geforce Titan X)
        """
        return self.entry['name']

    @property
    def index(self):
        """
        Returns the index of GPU card (e.g. Geforce Titan X)
        """
        return self.entry['index']
    
    @property
    def memory_total(self):
        """
        Returns the total memory (in MB) as an integer.
        """
        return int(self.entry['memory.total'])

    @property
    def memory_used(self):
        """
        Returns the occupied memory (in MB) as an integer.
        """
        return int(self.entry['memory.used'])

    @property
    def memory_available(self):
        """
        Returns the available memory (in MB) as an integer.
        """
        v = self.memory_total - self.memory_used
        return max(v, 0)

    @property
    def temperature(self):
        """
        Returns the temperature of GPU as an integer,
        or None if the information is not available.
        """
        v = self.entry['temperature.gpu']
        return int(v) if v is not None else None

    @property
    def utilization(self):
        """
        Returns the GPU utilization (in percentile),
        or None if the information is not available.
        """
        v = self.entry['utilization.gpu']
        return int(v) if v is not None else None

    def jsonify(self):
        o = dict(self.entry)
        o['processes'] = [{k: v for (k, v) in p.iteritems() if k != 'gpu_uuid'}
                          for p in self.processes]
        return o

class GPUManager(object):

    def __init__(self):
        self.gpu_list = GPUManager.new_query()
        
        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        
            

    @staticmethod
    def new_query():
        # 1. get the list of gpu and status
        gpu_query_columns = ('index', 'uuid', 'name', 'temperature.gpu',
                             'utilization.gpu', 'memory.used', 'memory.total')
        gpu_list = []

        smi_output = execute_process(
            r'nvidia-smi --query-gpu={query_cols} --format=csv,noheader,nounits'.format(
                query_cols=','.join(gpu_query_columns)
            ))

        for line in smi_output.split('\n'):
            if not line: continue
            query_results = line.split(',')

            g = GPUStat({col_name: col_value.strip() for
                         (col_name, col_value) in zip(gpu_query_columns, query_results)
                         })
            gpu_list.append(g)

        return gpu_list

    def update(self):
        self.gpu_list = GPUManager.new_query()
        
    # choose gpu by highest memory available
    # if available memory are same, choose lowest temperature
    def auto_choice(self, num=1):
        # sort by available memory
        
        # find index
        indices = np.lexsort(([gpu.temperature for gpu in self.gpu_list],
                              [-gpu.memory_available for gpu in self.gpu_list]))
        # num can not larger than all available gpu
        num = min(len(self.gpu_list), num)

        # set the environment
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in indices[:num])
        

if __name__=='__main__':
    import tensorflow as tf
    
    gpu_manager = GPUManager()    
    gpu_manager.auto_choice(num=2)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2)
    
    # use part of memory 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(sess.run([node1, node2]))
    
    gpu_manager.update()   
    for gpu in gpu_manager.gpu_list:
        print(gpu.jsonify())
    