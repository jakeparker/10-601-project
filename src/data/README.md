# CIFAR-10 Dataset

## File Types
* `*.mat` — data in matlab format.
* `*.bin` — data in binary format.

## Files
*  `labels.*` — gives meaningful names to the numeric labels.
*  `train.*` — 12000 training images and labels (4000 images per image class).
*  `test.*` — 3000 testing images (1000 images per image class).

## Data Format
*  `*.mat`
  *  `data` — a 12000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
  *  `labels` — a list of 12000 numbers in the range 0-2. The number at index i indicates the label of the ith image in the array data.


* `*.bin`
  *  `train.bin` — formatted as <1 x label><3072 x pixel>  
    *  the first byte is the label of the first image, which is a number in the range 0-2.
    *  The next 3072 bytes are the values of the pixels of the image.
    *  The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue.
    *  The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
    *  It contains 12000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 12000*3073 bytes long.
  *  `test.bin` — formatted as <3072 x pixel>
