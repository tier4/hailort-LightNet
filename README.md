# HailoRT LightNet for Low Power Edge AI


https://github.com/tier4/hailort-LightNet/assets/127905436/ec97e1b6-6db0-4a11-820d-2937eef7d4e2



## Purpose

This package implements LightNet inference on Hailo-8 chip with HailoRT for low power inference.
## Setup

Download HailoRT and HailoRT PCIe Driver from Hailo (https://hailo.ai/ja/developer-zone/)

```bash
sudo dpkg ­­install hailort_<version>_$(dpkg ­­print­architecture).deb hailort­-pcie­driver_<version>_all.deb
```


Moreover, you need to install as bellow.
```bash
sudo apt-get install libgflags-dev
```

## Building

```bash
git clone git@github.com:tier4/hailort-LightNet.git
cd hailort-LightNet
cd build/
cmake ..
make -j
```

## Start inference

-Infer from a Video

```bash
./hailort-lightnet --hef {HEF Name} --v {Video Name} 
```


## Parameters

--hef=HEF Path (Hailo Model)

--v=Video Path

--c=Number of Classes

--thresh=Score Thresh

--cam_id=camera id

## Assumptions / Known limits

## HEF model


### Todo

- [] Optimize Preprocess/Postprocess

## Reference repositories

- <https://github.com/daniel89710/lightNet>

- <https://github.com/daniel89710/lightNet-TRT>

