## **DeepFaceLab** is a tool that utilizes deep learning to recognize and swap faces in pictures and videos.

If you like this software, please consider a donation.

Goal: RTX 2080 TI

[Donate via Yandex.Money](https://money.yandex.ru/to/41001142318065)

[Donate via Paypal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=KK5ZCH4JXWMQS&source=url)

[Donate via Alipay](https://i.loli.net/2019/01/13/5c3ae3829809f.jpg)

bitcoin:31mPd6DxPCzbpCMZk4k1koWAbErSyqkAXr

### **Features**:

- new models

- new architecture, easy to experiment with models

- face data embedded to png files

- cpu mode. 8th gen Intel core CPU able to train H64 model in 2 days.

- new preview window

- extractor in parallel

- converter in parallel

- added **--debug** option for all stages

- added **MTCNN extractor** which produce less jittered aligned face than DLIBCNN, but can produce more false faces. Comparison dlib (at left) vs mtcnn on hard case:
![](https://i.imgur.com/5qLiiOV.gif)
MTCNN produces less jitter.

- added **Manual extractor**. You can fix missed faces manually or do full manual extract:
![](https://github.com/iperov/DeepFaceLab/blob/master/doc/manual_extractor_0.jpg)
![Result](https://user-images.githubusercontent.com/8076202/38454756-0fa7a86c-3a7e-11e8-9065-182b4a8a7a43.gif)

- standalone zero dependencies ready to work prebuilt binary for all windows versions, see below

### Warning: **Facesets** of FaceSwap or FakeApp are **not compatible** with this repo. You should to run extract again.

### **Model types**:

- **H64 (2GB+)** - half face with 64 resolution. It is as original FakeApp or FaceSwap, but with new TensorFlow 1.8 DSSIM Loss func and separated mask decoder + better ConverterMasked. for 2GB and 3GB VRAM model works in reduced mode.

H64 Robert Downey Jr.:

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/H64_Downey_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/H64_Downey_1.jpg)

- **H128 (3GB+)** - as H64, but in 128 resolution. Better face details. for 3GB and 4GB VRAM model works in reduced mode.

H128 Cage:

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/H128_Cage_0.jpg)

H128 asian face on blurry target:

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/H128_Asian_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/H128_Asian_1.jpg)

- **DF (5GB+)** - @dfaker model. As H128, but fullface model. Strongly recommended not to mix various light conditions in src faces.

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/DF_Cage_0.jpg)

- **LIAEF128 (5GB+)** - Less agressive Improved Autoencoder Fullface 128 model. Result of combining DF, IAE, + experiments. Model tries to morph src face to dst, while keeping facial features of src face, but less agressive morphing. Model has problems with closed eyes recognizing.

LIAEF128 Cage:

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/LIAEF128_Cage_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/LIAEF128_Cage_1.jpg)

LIAEF128 Cage video:

[![Watch the video](https://img.youtube.com/vi/mRsexePEVco/0.jpg)](https://www.youtube.com/watch?v=mRsexePEVco)

- **SAE ( minimum 2GB+, recommended 11GB+ )** - Styled AutoEncoder - new superior model based on style loss. SAE works as stylizer/morpher and does not guarantee that predicted face will look as src. Face obstructions also reconstructed without any masks. Converter mode 'overlay' should be used. Model has several options on start for fine tuning to fit your GPU. For more info read tips below.

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Asian_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Cage_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Cage_1.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Navalniy_0.jpg)

SAE model Cage-Trump video: https://www.youtube.com/watch?v=2R_aqHBClUQ

SAE model Putin-Navalny video: https://www.youtube.com/watch?v=Jj7b3mqx-Mw

Scene with extremely obstructed face in helmet, that cannot be handled by any other classic faceswap model (how to train it read tips):

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Cage_2.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/SAE_Cage_3.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/DeepFaceLab_convertor_overview.png)

### **Tips and tricks**:

unfortunately deepfaking is time/eletricpower consuming topic and has a lot of nuances.

Quality of src faceset significantly affects the final face.

Narrow src face is better fakeable than wide. This is why Cage is so popular in deepfakes.

Every model is good for specific scenes and faces.

H64 - good for straight faces as a demo and for low vram.

H128 - good for straight faces, gives highest resolution and details possible in 2019. Absolute best for asian faces, because they are flat, similar and evenly lighted with clear skin.

DF - good for side faces, but results in a lower resolution and details. Covers more area of cheeks. Keeps face unmorphed. Good for similar face shapes.

LIAE - can partially fix dissimilar face shapes, but results in a less recognizable face.

SAE tips:

- SAE - actually contains all other models, but better due to multiscale decoder + pixel loss. Just set style powers to 0.0 to work as default (H128/DF/LIAE) model.

- if src faceset has number of faces more than dst faceset, model can be not converged. In this case try 'Feed faces to network sorted by yaw' option.

- if src face wider than dst, model can be not converged. In this case try to decrease 'Src face scale modifier' to -5.

- architecture 'df' make predicted face looking more like src, but if model not converges try default 'liae'.

- if you have a lot of VRAM, you can choose between batch size that affects quality of generalization and enc/dec dims that affects image quality.

- how to train extremely obstructed face model with SAE:

First train it with both style powers at 10.0+ value. When artifacts become appearing at ~30-40k epochs, set face style to 0.0 or 0.01 and bg style to 0.1-0.3 and continue training. You can slightly vary theese values during training if something goes wrong. If the colors do not match, increase styles to 1.0 - 3.0. Experiment on your own. Track changes in preview history.

Improperly matched dst landmarks may significantly reduce fake quality:

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/Tips_improperly_dst_landmarks_0.jpg)

![](https://github.com/iperov/DeepFaceLab/blob/master/doc/Tips_improperly_dst_landmarks_1.jpg)

in this case watch "Manual re-extract bad dst aligned frames tutorial" below.

@GAN-er advanced tips:

Tip 1:
You may benefit by starting with a small batch size (within reason) and increasing it later. The reason is that a **large batch size will give you a more accurate descent direction but it will also be costlier to calculate**, and when you just start, you care mostly about the general direction; no need to sacrifice speed for precision at that point. There are plenty of sources discussing the batch size, as an example you can check this one:
https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network

Tip 2:
Unlike the batch size that the only thing that does is affecting how accurate each step will be as far a the true gradient goes, the dimensions, actually, increase the complexity of your NN. As a rule, **the more complex a network the better the resulting model**, but since nothing comes for free, **the more complex the network the more time it will take to converge**.
What you generally want is to **_figure out the max dimensions that you can use_** given your GPU's memory, and your desired max batch size.
You can set the max batch size to something, say K, and then increase the dimensions until you get OOM errors. In the end, you will end up with a triplet, {batch size, ae_dims, ed_dims}
Ideally, you would use 1024 and 85 for your autoencoder and encoder/decoder dimensions, but no card has enough memory for such a configuration even with batch size 1.
Remember that unlike batch size that you can change at will, once you set up the dimensions you can not change them.

Note that **if you use a complex - high number of dimensions NN, in combination with a small batch size, it will take _considerably_ longer for your model to converge**. So keep that in mind! You will simply have to wait longer, but also you will get a much much better result.

For cards with 11Gb of memory, and for SAE you can try the following settings:
For DF architecture: 12 698 51
For LIAEF architecture: 8 402 47

Tip 3:
If you end up being stuck, i.e. the loss does not go down but for no obvious reason or if you get weird artifacts in some previews before you discard and start from scratch, you may want to flip your DST and SRC for a while. This often is all you need to keep things going again.

Tip 4:
99.995% of your success or failure rate is due to bad SRC or DST sets. This means that 99.995% of your time should be spent in actually ensuring that your sets are well curated. Throwing together a hot podge of material and expecting a decent outcome is guaranteed to result in disappointment. Garbage in, garbage out.

### **Sort tool**:

`blur` places most blurred faces at end of folder

`hist` groups images by similar content

`hist-dissim` places most similar to each other images to end.

`hist-blur` sort by blur in groups of similar content

`brightness` 

`hue`

`black` Places images which contains black area at end of folder. Useful to get rid of src faces which cutted by screen.

`final` sorts by yaw, blur, and hist, and leaves best 1500-1700 images.

Best practice for gather src faceset from tens of thousands images:

1) `black` -> then delete faces cutted by black area at end of folder
2) `blur` -> then delete blurred faces at end of folder
3) `hist` -> then delete groups of similar unwanted faces and leave only target face
4) `final` -> then delete faces occluded by obstructions

Best practice for dst faces:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `hist` -> then delete groups of similar and leave only target face

### **Ready to work facesets**:

Nicolas Cage 4 facesets (1 mix + 3 different), Steve Jobs, Putin

download from here: https://mega.nz/#F!y1ERHDaL!PPwg01PQZk0FhWLVo5_MaQ

### Video tutorials for prebuilt windows binary

Basic workflow: https://www.youtube.com/watch?v=K98nTNjXkq8

Basic workflow (derpfakes): https://www.youtube.com/watch?v=cVcyghhmQSA

Manual re-extract bad dst aligned frames: https://www.youtube.com/watch?v=7z1ykVVCHhM

### **Build info**

dlib==19.10.0 from pip compiled without CUDA. Therefore you have to compile DLIB manually, orelse use MT extractor only.

Command line example for windows: `python setup.py install -G "Visual Studio 14 2015" --yes DLIB_USE_CUDA`

### **CPU only mode**

CPU mode enabled by arg --cpu-only for all stages. Follow requirements-cpu.txt to install req packages.
Do not use DLIB extractor in CPU mode, its too slow.
Only H64 model reasonable to train on home CPU.

### Mac/linux/docker script support.

This repo supports only windows build of scripts. If you want to support mac/linux/docker - create fork, it will be referenced here.

### Prebuilt windows app:

Windows 7,8,8.1,10 zero dependency (just install/update your GeForce Drivers) prebuilt DeepFaceLab (include GPU and CPU versions) can be downloaded from 
1) torrent https://rutracker.org/forum/viewtopic.php?p=75318742 (magnet link inside).
2) https://mega.nz/#F!b9MzCK4B!zEAG9txu7uaRUjXz9PtBqg

### Communication groups:

(Chinese) QQ group 951138799 for ML/AI experts

[deepfakes (Chinese)](https://deepfakes.com.cn/)

[mrdeepfakes (English)](https://mrdeepfakes.com/forums/)

[reddit (English)](https://www.reddit.com/r/GifFakes/new/)