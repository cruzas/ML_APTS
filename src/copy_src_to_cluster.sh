#!/bin/bash -l

rsync -azv -e 'ssh daint' ../src/ :/scratch/snx3000/scruzale/MultiscAI/ML2/src/
