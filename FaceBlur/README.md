# FaceBlur
Face detection and blurring for of multiple images using opencv and face_recognition. Currently I use the simple methods provided by the [face_recognition](https://github.com/ageitgey/face_recognition) (wrapper) library. Given the coordinates of the bounding boxes i use opencv filters to blur the image parts. All this is done with some degree of parallelism and python3s standard library for mp.      
The [face_recognition](https://github.com/ageitgey/face_recognition) however, still has trouble finding some (conspicuous) faces, although it is supposed to have an accuracy of  99.38% with [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/). Maybe i will add additional ML techniques or face rec libraries to increase the accuracy.   
For my tests i mostly used images after googeling for images about "protests"


## Requirements
* opencv
* face_recognition

## Usage:    
* `python3 FaceBlur.py imgs` where imgs is the folder containing images to be blurred
