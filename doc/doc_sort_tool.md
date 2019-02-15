### **Sort tool**:

Extraction is rarely perfect and a final pass by human eyes is typically required. This is the best way to remove unmatched or misaligned faces quickly. The sort tool included in the project greatly reduces the time and effort required to clean large sets. Like pictures will be grouped together and false positives can be quickly be identified.

`blur` places most blurred faces at end of folder

`hist` groups images by similar content

`hist-dissim` places most dissimilar to each other images to begin.

`hist-blur` sort by blur in groups of similar content

`face-pitch` sort by face pitch direction

`face-yaw` sort by face yaw direction

`brightness` 

`hue`

`black` Places images which contains black area at end of folder. Useful to get rid of src faces which cutted by screen.

`final` sorts by yaw, blur, and hist, and leaves best 1500-1700 images.

Suggested sort workflow for gathering src faceset from very large image pools:

1) `black` -> then delete faces cutted by black area at end of folder
2) `blur` -> then delete blurred faces at end of folder
3) `hist` -> then delete groups of similar unwanted faces and leave only target face
4) `final` -> then delete faces occluded by obstructions

Suggested sort workflow for cleaning dst faceset:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `hist` -> then delete groups of similar and leave only target face
