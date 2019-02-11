### **Tips and tricks**:

unfortunately deepfaking is time/eletricpower consuming topic and has a lot of nuances.

Quality of src faceset significantly affects the final face.

Narrow src face is better fakeable than wide. This is why Cage is so popular in deepfakes.

Every model is good for specific scenes and faces.

H64 - good for straight faces as a demo and for low vram.

H128 - good for straight faces, gives higher resolution and details.

DF - good for side faces, but results in a lower resolution and details. Covers more area of cheeks. Keeps face unmorphed. Good for similar face shapes.

LIAE - can partially fix dissimilar face shapes, but results in a less recognizable face.

SAE - new flexible model. Absolute best in 2019.

SAE tips:

- SAE - actually contains all other models, but better due to smooth DSSIM-MSE(pixel loss) transition. Just set style powers to 0.0 to work as default (H128/DF/LIAE) model.

- if src faceset has number of faces more than dst faceset, model can be not converged. In this case try 'Feed faces to network sorted by yaw' option.

- if src face wider than dst, model can be not converged. In this case try to decrease 'Src face scale modifier' to -5.

- default architecture 'df' make predicted face looking more like src, but if model not converges try 'liae'.

- if you have a lot of VRAM, you can choose between batch size that affects quality of generalization and enc/dec dims that affects image quality.

- common training algorithm for styled face: set initial face and bg style values to 10.0, train it to 15k-20k epochs, then overwrite settings and set face style to 0.1, bg style to 4.0, and train it up to clear result.

- how to train extremely obstructed face model with SAE? First train the styled model on clean dst faces without obstructions. Then start new training on your target video, save it on 1+ epoch, replace model files with pretrained model and continue training. Experiment with styling values on your own during training. Enable 'write preview history' and track changes. Backup model files every 10k epochs. You can revert model files and change values if something goes wrong.

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
