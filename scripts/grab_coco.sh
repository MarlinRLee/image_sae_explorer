#!/bin/bash -l        
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=lee02328@umn.edu
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=32GB                    # Memory per node

cd /scratch.global/lee02328
mkdir -p coco && cd coco
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -o '*.zip' && rm -f *.zip