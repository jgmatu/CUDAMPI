#!/bin/bash

mpirun -mca btl ^openib -np 4 ./box_filter NASA_Mars_Rover.jpg
