#!/bin/bash

mpirun -mca btl ^openib -np 1 ./master : -np 2 ./slave
