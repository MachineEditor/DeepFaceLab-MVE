### **Features**:

- standalone zero dependencies ready to work prebuilt binary for all windows versions, see below

- new models

- new architecture, easy to experiment with models

- face data embedded to JPG files

- cpu mode. 8th gen Intel core CPU able to train H64 model in 2 days.

- new preview window

- extractor in parallel

- converter in parallel

- **--debug** option for all stages

- **MTCNN extractor** which produce less jittered aligned face than DLIBCNN, but can produce more false faces. Comparison dlib (at left) vs mtcnn on hard case:
![](https://i.imgur.com/5qLiiOV.gif)
MTCNN produces less jitter.

- **Manual extractor**. You can fix missed faces manually or do full manual extract:
![](manual_extractor_0.jpg)
![Result](https://user-images.githubusercontent.com/8076202/38454756-0fa7a86c-3a7e-11e8-9065-182b4a8a7a43.gif)
