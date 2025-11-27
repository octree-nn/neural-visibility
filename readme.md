## Neural Visibility of Point Sets

This repository contains the implementation of **Neural Visibility of Point Sets**.


**[Neural Visibility of Point Sets](http://arxiv.org/abs/2509.24150)**<br/>
Jun-Hao Wang, Yiyang Tian, [Baoquan Chen](https://baoquanchen.info/), [Peng-Shuai Wang](https://wang-ps.github.io/)</br>
Accepted by SIGGRAPH Asia 2025
![teaser](assets/teaser.png)


- [Neural Visibility of Point Sets](#neural-visibility-of-point-sets)
  - [1. Installation](#1-installation)
  - [2. Training](#2-training)
  - [3. UI](#3-ui)
  - [4. Citation](#4-citation)


### 1. Installation

This code has been tested on Ubuntu 20.04 with 4 Nvidia 4090 GPUs (24GB memory).

1.1 Install Conda and create a Conda environment

   ```bash
   conda create --name neuvis python=3.13
   conda activate neuvis
   ```

1.2 Clone this repository and Install the required packages.

   ```bash
   git clone https://github.com/Timekisser/NeuVis.git
   conda install ocnn thsolver tqdm tensorboard
   ```

1.3 Install PyTorch=2.6.0 with conda according to the official documentation.

   ```
   pip3 install torch torchvision torchaudio
   ```

### 2. Training

2.1 Download the [ShapeNetV2](https://shapenet.org/) dataset and extract it to the data directory. You could also use other dataset, as long as the folder is organized as:

   ```
   --ShapeNet
   	--class1
   		--model1
            --model_normalized.obj
            ...
   		--model2
            --model_normalized.obj
            ...
   		...
   	--class2
   	...
   ```

   Then run the script:
   ```
   python prepare_data.py --in_path [inpath] --out_path [outpath]
   ```
   We provide the filelist of ShapeNetV2 in the `data` folder, which is defaultly used as the `--out_path` in the above command. If you want to use other datasets, please generate the filelist by yourself.
   
2.2 Run the training script

   ```
   python train.py --config vis_shapenet.yaml SOLVER.gpu 0,1,2,3
   ```

2.3 Evaluate the model by:
   ```
   python train.py --config vis_shapenet.yaml SOLVER.gpu 0, SOLVER.run test DATA.test.takes [numofpoints]
   ```
   If you want to evaluate on each category on ShapeNet, you could run the script:
   ```
   python script.py
   ```
   See the `script.py` for more details.
   We also provide our pretrained model at [Huggingface](https://huggingface.co/JKAlice/neuvis). You can download and put it in the `logs/depth8/checkpoints` folder, then run the script above to evaluate it.

### 3. UI

3.1 Download our pretrained model at [Huggingface](https://huggingface.co/JKAlice/neuvis).

3.2 Install the required packages.

   ```
   pip install -r requirements.txt
   ```

3.3 Prepare the data as .npz file.

   We provide a script to convert the .obj files to .npz files. Run it by: 

   ```
   python sample.py --name [path-to-obj-file] --num_pts [number-of-points] 
   ```

3.4 Run the UI script.

   ```
   python ui.py --model_path [path-to-model-weights]
   ```

   You could drag and drop the .npz file to the UI window to visualize the result.

### 4. Citation
 
    ```bibtex
    @inproceedings{wang2025neural,
      title={Neural Visibility of Point Sets},
      author={Wang, Jun-Hao and Tian, Yiyang and Chen, Baoquan and Wang, Peng-Shuai},
      booktitle={SIGGRAPH Asia},
      year={2025},
    }
      