from PIL import Image
import numpy as np

def ImageShrink2(image):
    # print(image.size)
    small_image = image.resize((image.size[0] // 2, image.size[1] // 2))
    return small_image

def ComputeBitmaps(image):
    image_np = np.array(image)
    upper = np.percentile(image_np, 54)
    lower = np.percentile(image_np, 46)
    mid = np.percentile(image_np, 50)
    col, row = image.size
    tb = image_np > mid
    eb = (image_np > upper) | (image_np < lower)
    # for i in range(row):
    #     for j in range(col):
    #         if image_np[i, j] > mid:
    #             tb[i, j] = 1
    #         if lower < image_np[i, j] and image_np[i, j] < upper:
    #             eb[i, j] = 0
    return (tb, eb)

def BitmapShift(row, col, bm, xo, yo):
    # shifted_bm = np.zeros((row, col), dtype=bool)
    # for i in range(row):
    #     for j in range(col):
    #         shifted_x = i + xo
    #         shifted_y = j + yo
    #         if ((0 <= shifted_x and shifted_x < row) and (0 <= shifted_y and shifted_y < col)) and bm[i, j] == 1:
    #             shifted_bm[shifted_x, shifted_y] = 1
    shift_bm = np.roll(bm, xo, axis = 0)
    shift_bm = np.roll(shift_bm, yo, axis = 1)
    if xo > 0:
        shift_bm[0:xo,] = False
    elif xo < 0:
        shift_bm[row + xo:row,] = False
    
    if yo > 0:
        shift_bm[:, 0:yo] = False
    elif yo < 0:
        shift_bm[:,col + yo:col] = False
    
    return shift_bm

def BitmapXOR(bm1, bm2):
    result = np.bitwise_xor(bm1, bm2)
    return result

def BitmapAND(bm1, bm2):
    result = np.bitwise_and(bm1, bm2)
    return result

def BitmapTotal(bm):
    return np.sum(bm)

def GetExpShift(image1, image2, shift_bits, shift_ret):
    cur_shift = [0, 0]
    if shift_bits > 0:
        sml_img1 = ImageShrink2(image1)
        sml_img2 = ImageShrink2(image2)
        GetExpShift(sml_img1, sml_img2, shift_bits-1, cur_shift)
        # free?
        cur_shift[0] *= 2
        cur_shift[1] *= 2
    else:
        cur_shift[0] = cur_shift[1] = 0
    tb1, eb1 = ComputeBitmaps(image1)
    tb2, eb2 = ComputeBitmaps(image2)
    min_err = image1.size[0] * image1.size[1]
    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = cur_shift[0] + i
            ys = cur_shift[1] + j
            shifted_tb2 = BitmapShift(image1.size[1], image1.size[0], tb2, xs, ys)
            shifted_eb2 = BitmapShift(image1.size[1], image1.size[0], eb2, xs, ys)
            diff_b = BitmapXOR(tb1, shifted_tb2)
            diff_b = BitmapAND(diff_b, eb1) 
            diff_b = BitmapAND(diff_b, shifted_eb2)
            err = BitmapTotal(diff_b)
            # print(err)
            if err < min_err:
                shift_ret[0] = xs
                shift_ret[1] = ys
                min_err = err
            # free?
    # free?
image1 = Image.open("?.jpg") # 這個填第一張照片檔名 (以這張為基準把其他都跟他對齊)
image1L = image1.convert("L")
image2 = Image.open("?.jpg") # 這個填第二張照片檔名 (要被shift的對象)
image2L = image2.convert("L")
shift_ret = [0, 0]
GetExpShift(image1L, image2L, 6, shift_ret)
new_Image2 = image2.transform(image2.size, Image.Transform.AFFINE, data=(1,0,-shift_ret[1],0,1,-shift_ret[0]))
new_Image2.save("??.jpg") # 這個填第二張照片被shift過後的結果
