### **Features**:

- Windows binary containing pre-compiled dependencies, including CUDA libraries.

- New models expanding upon the original faceswap model.

- Model architecture designed with experimentation in mind.

- Face metadata embedded into extracted JPG files.

- CPU-only mode [`--cpu-mode`]. 8th gen Intel core CPU able to train H64 model in 2 days.

- Preview window

- Extractor and Converter run in parallel.

- Debug mode option for all stages: [`--debug`]

- Multiple extraction modes: MTCNN, dlib, or manual.

#### Extractor Examples
##### MTCNN

Predicts faces more uniformly than dlib, resulting in a less jittered aligned output. However, MTCNN extraction will produce more false positives.


Comparison dlib (at left) vs mtcnn on hard case:
![](https://i.imgur.com/5qLiiOV.gif)

- **Manual Extractor**

A manual extractor is available. This extractor uses the preview GUI to allow the user to properly align detected faces. 

![](manual_extractor_0.jpg)

This mode can also be used to fix incorrectly extracted faces. Manual extraction can be used to greatly improve training on face sets that are heavily obstructed. 

![Result](https://user-images.githubusercontent.com/8076202/38454756-0fa7a86c-3a7e-11e8-9065-182b4a8a7a43.gif)

