# Traffic Vehicle Counter

This code is desgined to use object detection + tracking algorithms to track and count vehicles on videos.

Implemented Object Detectors:
- YOLOv5 (S, M, L, X)
- YOLOv3 (tiny, yv3, yv3-spp)

Implemented Tracking Algorithms:
- SORT


# Install
- Donwload this repo source code, via `git clone` or zip file (and unzip it).

- You need to have Python. Install it from [Python Website](https://www.python.org/) or [Anaconda](https://anaconda.org/), for example.

- Then, use your package manager to create a virtual environment *envname* with Python version greather or equal to 3.9.0, cd to the source code folder and pip install requirements.txt, such as:

```bash
conda create --name envname python=3.9.0

conda activate envname

cd source_code_folder_path

pip install -r requirements.txt
```

# How to Use
The easy way: jupyter notebooks! Open yout terminal and:

```
conda activate envname

jupyter notebook
```

Then, navigate to the **interface_traffic_vehicle_counter.ipynb** and run it!

To define the counting barriers, there is two jupyter notebooks to help doing it: 

- **interface_define_barriers_jupyter**: Define barriers just via browser, it's nice if you use remote location server.

- **interface_define_barriers_opencv**: Define barrier using opencv package and works only for local machine. It's faster than the *interface_define_barriers_jupyter*.

If yout want to use via command line, please refer to the source code.

# Research
-  https://github.com/YuriRibeiro/cfd

