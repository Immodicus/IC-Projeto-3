# Information and codification (IC) - Project 3

[![Build Linux](https://github.com/Immodicus/IC-Projeto-3/actions/workflows/build-linux.yml/badge.svg?branch=master)](https://github.com/Immodicus/IC-Projeto-3/actions/workflows/build-linux.yml)

## Group members - G11

|  NMec | Name                | email                   |
| ----: | ------------------- | ----------------------- |
| 97147 | Jodionísio Muachifi | jodionsiomuachifi@ua.pt |
| 97688 | Rúben Castelhano    | rubencastelhano@ua.pt   |
| 93413 | Tiago Rodrigues     | rodriguestiago@ua.pt    |

## Building:

- git clone git@github.com:Immodicus/IC-Projeto-3.git
- cd IC-Projeto-3
- mkdir build && cd build
- cmake -DCMAKE_BUILD_TYPE=Release ..
- make

## Testing:
**Note:** You can run every program with no arguments to see a complete list of all possible arguments and their use.

### Ex.1:
```bash
video_codec -i [-d] fileIn fileOut
```
#### Example:
```bash
./video_codec -i akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded
./video_codec -i -d encoded akiyo_decoded.y4m # decompress encoded to akiyo_decoded.y4m
```
### Ex.2:
```bash
video_codec [-d] -bs blockSize -sa searchArea -kf keyFrameInterval fileIn fileOut
```
#### Example:
```bash
./video_codec -bs 16 -sa 8 -kf 32 akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded
./video_codec -d encoded akiyo_decoded.y4m # decompress encoded to akiyo_decoded.y4m
```

### Ex.3:
```bash
video_codec [-d] -bs blockSize -sa searchArea -kf keyFrameInterval -q numShift -fq numBits fileIn fileOut
```
```bash
./video_cmp originalFile decompressedFile
```

#### Example:
```bash
./video_codec -bs 16 -sa 8 -kf 32 -q 2 akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded discarding last 2 bits of residuals
./video_codec -bs 16 -sa 8 -kf 32 -fq 4 akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded quantizing samples to 4 bits before prediction
./video_codec -d encoded akiyo_decoded.y4m # decompress encoded to akiyo_decoded.y4m
```
```bash
./video_cmp akiyo_cif.y4m akiyo_decoded.y4m
```

### Ex.4:
```bash
video_codec_dct [-d] -bs blockSize -sa searchArea -kf keyFrameInterval -q quality fileIn fileOut
```
#### Example:
```bash
./video_codec_dct -i -q 90 akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded using only DCT 2D intra-frame
./video_codec_dct -bs 16 -sa 8 -kf 32 -q 90 akiyo_cif.y4m encoded # compress akiyo_cif.y4m to encoded using DCT 2D intra-frame and motion compensation
./video_codec_dct -d encoded akiyo_decoded.y4m # decompress encoded to akiyo_decoded.y4m
```
