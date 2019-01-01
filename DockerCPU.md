# For Mac Users
If you just have a **MacBook**.DeepFaceLab **GPU** mode does not works. However,it can also works with **CPU** mode.Follow the Steps below will help you build the **DRE** (DeepFaceLab Runtime Environment) Easier.

### 1. Open a new terminal and Clone DeepFaceLab with git
```
$ git git@github.com:iperov/DeepFaceLab.git
```

### 2. Change the directory to DeepFaceLab
```
$ cd DeepFaceLab
```

### 3. Install Docker

[Docker Desktop for Mac](https://hub.docker.com/editions/community/docker-ce-desktop-mac)

### 4. Build Docker Image For DeepFaceLab

```
$ docker build -t deepfacelab-cpu -f Dockerfile.cpu .
```

### 5. Mount DeepFaceLab volume and Run it

```
$ docker run -p 8888:8888  --hostname deepfacelab-cpu --name deepfacelab-cpu  -v $PWD:/notebooks  deepfacelab-cpu
```

PS: Because your current directory is `DeepFaceLab`,so `-v $PWD:/notebooks` means Mount `DeepFaceLab` volume to `notebooks` in **Docker**

And then you will see the log below:

```
The Jupyter Notebook is running at:
http://(deepfacelab-cpu or 127.0.0.1):8888/?token=your token
```

### 6. Open a new terminal to run DeepFaceLab in /notebooks

```
$ docker exec -it deepfacelab-cpu bash
$ ls -A
```

### 7. Use jupyter in deepfacelab-cpu bash

```
$ jupyter notebook list
```
or just open it on your browser `http://127.0.0.1:8888/?token=your_token`

PS: You can run python with jupyter.However,we just run our code in bash.It's simpler and clearer.Now the **DRE** (DeepFaceLab Runtime Environment) almost builded.

### 8. Stop or Kill Docker Container

```
$ docker stop deepfacelab-cpu
$ docker kill deepfacelab-cpu
```

### 9. Start Docker Container

```
# start docker container
$ docker start deepfacelab-cpu
# open bash to run deepfacelab
$ docker exec -it deepfacelab-cpu bash
```

PS: `STEP 8` or `STEP 9` just show you the way to stop and start **DRE**.

### 10. enjoy it

```
# make sure you current directory is `/notebooks`
$ pwd
# make sure all `DeepFaceLab` code is in current path `/notebooks`
$ ls -a
# read and write permission
$ chmod +x cpu.sh
# run `DeepFaceLab`
$ ./cpu.sh
```

### Details with `DeepFaceLab`

#### 1. Concepts

![SRC](doc/DF_Cage_0.jpg)

In our Case,**Cage**'s Face is **SRC Face**,and **Trump**'s Face is **DST Face**.and finally we get the **Result** below.

![Result](doc/merged-face.jpg)

So,before you run `./cpu.sh`.You should be aware of this.

#### 2. Use MTCNN(mt) to extract faces
Do not use DLIB extractor in CPU mode

#### 3. Best practice for SORT
1) delete first unsorted aligned groups of images what you can to delete.

2) use `hist`

#### 4. Use `H64 model` to train and convert
Only H64 model reasonable to train on home CPU.You can choice other  model like **H128 (3GB+)** | **DF (5GB+)** and so on ,it depends entirely on your CPU performance.

#### 5. execute the script below one by one

```
root@deepfacelab-cpu:/notebooks# ./cpu.sh
1) clear workspace		      7) data_dst sort by hist
2) extract PNG from video data_src    8) train
3) data_src extract faces	      9) convert
4) data_src sort		     10) converted to mp4
5) extract PNG from video data_dst   11) quit
6) data_dst extract faces
Please enter your choice:       
```

#### 6. Put all videos in `workspace` directory
```
.
├── data_dst
├── data_src
├── dst.mp4
├── model
└── src.mp4

3 directories, 2 files
```
