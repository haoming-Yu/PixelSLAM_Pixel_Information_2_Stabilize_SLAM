# Point-SLAM ++
1. This repo is used to implement my ECCV paper's code
2. This repo is only a temporary code repo. As for the final implementation, it will be released soon.
3. This repo need to first clean the repo of point-slam's work.
4. Format reference: https://github.com/Mael-zys/T2M-GPT, the format does not strictly follow the format.

# Log
## 2024-1-10
1. I've already clean all the code and test the running result of original code. In order to make sure it will work, I read all the codes.
2. Now we need to do the pruning strategy implementation, and point adding strategy based on the Gaussian implementation to improve the rendering speed and result at the same time.
3. How to run:
    
    Example: `python run.py configs/Replica/office0.yaml`
    
    If you want to run other scene, you can use different scene in the configs file.

4. Datasets:
    
    The datasets need to be downloaded. In the original paper, the evaluation is done on three datasets: Replica, TUM_RGBD, ScanNet. Here I didn't get proper data in ScanNet. So for now I've only tested on Replica and TUM_RGBD. Replica is the most important dataset in this task. And TUM_RGBD dataset remains to have many problems.

    Datasets creation: (you should do this before running)
    1. Download Replica
        
        `mkdir -p datasets`

        `cd datasets`
        
        `# you can also download the Replica.zip manually through`
        
        `# link: https://caiyun.139.com/m/i?1A5Ch5C3abNiL password: v3fY (the zip is split into smaller zips because of the size limitation of caiyun)`
        
        `wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip`
        
        `unzip Replica.zip`
    2. Download RUM_RGBD
        
        `mkdir -p datasets/TUM_RGBD`
        
        `cd datasets/TUM_RGBD`
        
        `wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz`
        
        `tar -xvzf rgbd_dataset_freiburg1_desk.tgz`
        
        `wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz`
        
        `tar -xvzf rgbd_dataset_freiburg1_desk2.tgz`
        
        `wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz`
        
        `tar -xvzf rgbd_dataset_freiburg1_room.tgz`
        
        `wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz`
        
        `tar -xvzf rgbd_dataset_freiburg2_xyz.tgz`
        
        `wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz`
        
        `tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz`


