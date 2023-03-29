# VFX Project 1 - HDR Imaging

## 0. Team Members
* B09902001 李奕辰
* R07922100 林峻毅

* Run

```
python3 hdr.py 
```
It will use pictures in 'pictures' and output the radiance map, response cureve and result.hdr to 'hdr', and the tonemapping results to 'tonemapping'

# Works

## Image Alignment

### MTB methods

## HDR Reconstruction

### Debevecs methods (with lamdas = 10)

## Tone mapping

### local tonemapping methods in 2002

### global tonemapping methods in 2002

### global tonemapping methods in 2005

# Results
## Radiance map 
| Blue | Green | Red |
| :--------------------------: | :-------------------------: | :-------: |
| ![](hdr/radiance_map_blue.png) | ![](hdr/radiance_map_green.png) | ![](hdr/radiance_map_red.png) |
## Response Curve
![](hdr/res_curve.png)
## tonemapping
| global_2002 | local_2002 | global_2005 |
| :--------------------------: | :-------------------------: | :-------: |
| ![](tonemapping/global_2002.png) | ![](tonemapping/local_2002.png) | ![](tonemapping/global_2005.png) |
