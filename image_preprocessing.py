# -*- coding: utf-8 -*-
from pathlib import Path
import ttkbootstrap as ttk
import tkinter as tk
from tkinter import filedialog
# from tkinter import *
from ttkbootstrap.constants import *
from ttkbootstrap.style import Bootstyle
import cv2
import numpy as np
import os
import scipy.signal
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
import ctypes
# import img_preprocessing_fns as fns
from img_preprocessing_fns import *
from tkinter import font as tkFont
from PIL import ImageGrab

#設定圖片路徑在assets資料夾
IMG_PATH = Path(__file__).parent / 'assets'

gui_width = 1675
gui_height = 950
ip_file = ""
op_file = ""
original_img = None
modified_img = None
result_modified_img_list = []
result_image_show = {}
result_image_show_label = {}
result_image_show_popp = {}
file_name_default2_list = []

# 儲存已選取的方法
def show_select_list():
    save_fns = []
    global imga
    imga = mpimg.imread(ip_file)
    
    for key,value in tk_var_dict.items():
        if value.get() == True:
            save_fns.append(group_list[key])
        else:
            pass

        # print(save_fns)    
    result = fns_muti_method(img=imga,method=save_fns)
        
    
#載入圖片
def load_file():
    global ip_file
    ip_file = filedialog.askopenfilename(
        title="Open an image file",
        initialdir=".",
        filetypes=[("All Image Files", "*.*")],
    )
    draw_before_canvas()
    for key,value in tk_var_dict.items():
        tk_var_dict[key].set(False)
    
    after_canvas.img = ""
        
    # print(f"Image loaded from: {ip_file}")

#儲存影像處理後的圖片
def save_file():
    global file_name_default
    
    file_name_default=""
    
    for key,value in tk_var_dict.items():
        
        if value.get() == True:
            file_name_default += fns_name_list[key] + "&"
    
    file_name_default = file_name_default[:-1]
    
    global ip_file, original_img, modified_img
    file_ext = os.path.splitext(ip_file)[1][1:]
    
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        initialfile=file_name_default
    )
    
    modified_img = modified_img.convert("RGB")
    modified_img.save(op_file)
    # print(f"Image saved at: {op_file}")

#顯示原始圖片，在GUI左側畫布上
def draw_before_canvas():
    global original_img, ip_file
    original_img = Image.open(ip_file)
    original_img = original_img.convert("RGB")
    img = ImageTk.PhotoImage(original_img)
    before_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    before_canvas.img = img

#顯示影像處理後的圖片，在GUI右側畫布上
def draw_after_canvas(mimg):
        
    global modified_img

    modified_img = Image.fromarray(mimg)
    # modified_img = Image.fromarray(np.uint8(mimg))

    img = ImageTk.PhotoImage(modified_img)
    print(img)
    after_canvas.create_image(
        256,
        256,
        image=img,
        anchor="center",
    )
    after_canvas.img = img
    
def open_popup_input(index_num):
    
    resizepopimg = result_modified_img_list[index_num].resize((600, 600))
    popimag = ImageTk.PhotoImage(resizepopimg)
    
    global popup
    popup = tk.Toplevel(root)
    popup.resizable(False, False)
    popup.title("Image")
    
    text_label = tk.Label(popup, image=popimag, justify="center")
    text_label.pack(side="top", anchor="center", padx=15, pady=10)
    popup.geometry(f"600x{50+text_label.winfo_reqheight()}")
    popup.mainloop()
    
   
# 應用多種影像處理方法
def fns_muti_method(img, method=[]):
    
    for i in range(len(method)):  
        img = method[i](img)
    # 將影像處理結果繪製在右側畫布
    draw_after_canvas(img)
    


def result_img_save():
    global file_name_default2
    
    file_name_default2 = ""
            
    for key,value in tk_var_dict.items():
        if value.get() == True:
            # file_name_default2 += fns_name_list[key] + "&"
            file_name_default2 += str(key + 1) + " & "
            
    file_name_default2_list.append(file_name_default2[:-2])
   
        
    result_modified_img_list.append(modified_img)
    col=1 # start from column 1
    row=3 # start from row 3
    for index,element in enumerate(result_modified_img_list):
        # print(len(result_modified_img_list))
        if(col<8):
                    
            resize_image = element.resize((200, 200))
            result_image_file = ImageTk.PhotoImage(resize_image)
            
            result_image_show[index] = tk.Label(result_canvas)
            result_image_show[index].grid(row=row, column=col)
            result_image_show[index].image = result_image_file
            result_image_show[index]['image']=result_image_file
            # result_image_show[index]['text']=file_name_default2_list[index]
            
            
            result_image_show_label[index] = tk.Label(result_canvas, text=file_name_default2_list[index])
            result_image_show_label[index].grid(row=row-1, column=col)
            
            result_image_show_popp[index] = tk.Button(result_canvas, text='Show Image(600x600)', command=lambda c=index: open_popup_input(c))
            result_image_show_popp[index].grid(row=row-2, column=col)
            
            col=col+1
            
     
def mutli_save_file():
    
    result_len = len(result_modified_img_list)
    
    fig, ax = plt.subplots(1,result_len,figsize = (50,50))
    
    for index,element in enumerate(result_modified_img_list):
        
        ax[index].imshow(element)
        ax[index].title.set_text(file_name_default2_list[index])
    
    plt.show()
    
    
    file_ext = os.path.splitext(ip_file)[1][1:]
    
    op_file = filedialog.asksaveasfilename(
        filetypes=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        defaultextension=[
            (
                f"{file_ext.upper()}",
                f"*.{file_ext}",
            )
        ],
        initialfile="result_preprocessing_images"
    )
    
    fig.savefig(op_file, dpi=100)
    
    
def result_img_reset():

    global result_canvas, buttom_frame, mutli_save_btn, reset_btn
    result_canvas.destroy()
    result_canvas = tk.Canvas(buttom_frame, bg="white", width=1265, height=220)
    result_canvas.pack(fill="both", expand=1, padx=20)
    
    mutli_save_btn.destroy()
    mutli_save_btn = ttk.Button(buttom_frame, text="Save", command=mutli_save_file)
    mutli_save_btn.pack(side='right')
    
    reset_btn.destroy()
    reset_btn = ttk.Button(buttom_frame, text="Reset", style='my.TButton', command=result_img_reset)
    reset_btn.pack(side='right', padx=15)
    
    global result_modified_img_list
    global result_image_show
    global result_image_show_label
    global result_image_show_popp
    global file_name_default2_list
    
        
    result_modified_img_list = []
    result_image_show = {}
    result_image_show_label = {}
    result_image_show_popp = {}
    file_name_default2_list = []

def reset_after_canvas():
    for key,value in tk_var_dict.items():
        tk_var_dict[key].set(False)
        
    after_canvas.img = ""

                         
    
if __name__ == '__main__':
    
    
    #gui介面基礎設定
    root = ttk.Window(
        title="Image processing",
        themename="cyborg",
        size=(gui_width, gui_height),
        resizable=(False, False),
    )
    
    #自訂義 Checkbutton 文字font樣式
    s = ttk.Style()
    s.configure('my.TCheckbutton', font=('Helvetica', 14))
    
    s1 = ttk.Style()
    s1.configure('my.TButton', foreground = 'white', background='red')
    
    #整個介面布局設定
    # frames
    buttom_frame = ttk.LabelFrame(root, text="Result Image", labelanchor="n")
    buttom_frame.pack(fill="x", side="bottom", padx=10, pady=10, expand=1, anchor="s")
    
    left_frame = ttk.LabelFrame(root, text="Original Image", labelanchor="n")
    left_frame.pack(fill="y", side="left", padx=10, pady=10, expand=1)
    
    middle_frame = ttk.LabelFrame(root, text="Algorithms", labelanchor="n")
    middle_frame.pack(fill="y", side="left", padx=5, pady=10)
    
    right_frame = ttk.LabelFrame(root, text="Modified Image", labelanchor="n")
    right_frame.pack(fill="y", side="left", padx=10, pady=10, expand=1)
    

    
    # bottom frame contents
    result_canvas = tk.Canvas(buttom_frame, bg="white", width=1265, height=220)
    result_canvas.pack(fill="both", expand=1, padx=20)
    mutli_save_btn = ttk.Button(buttom_frame, text="Save", command=mutli_save_file)
    mutli_save_btn.pack(side='right')
    
    reset_btn = ttk.Button(buttom_frame, text="Reset", style='my.TButton', command=result_img_reset)
    reset_btn.pack(side='right', padx=15)
    

    
    # left frame contents
    before_canvas = tk.Canvas(left_frame, bg="white", width=520, height=560)
    before_canvas.pack(expand=1, padx=20)
    
    browse_btn = ttk.Button(left_frame, text="Browse", command=load_file)
    browse_btn.pack(side='left')
    

    # middle frame contents
    algo_canvas = tk.Canvas(middle_frame, width=360, highlightthickness=0)
    scrollable_algo_frame = ttk.Frame(algo_canvas)
    scrollbar = tk.Scrollbar(
        middle_frame, orient="vertical", command=algo_canvas.yview, width=15
    )
    scrollbar.pack(side="right", fill="y")
    algo_canvas.pack(fill="both", expand=1, padx=20)
    algo_canvas.configure(yscrollcommand=scrollbar.set)
    algo_canvas.create_window((0, 0), window=scrollable_algo_frame, anchor="nw")
    scrollable_algo_frame.bind(
        "<Configure>", lambda _: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
    )
    
    
    # right frame contents
    after_canvas = tk.Canvas(right_frame, bg="white", width=520, height=560)
    after_canvas.pack(expand=1, padx=20)
    
    save_btn = ttk.Button(right_frame, text="Save", command=save_file)
    save_btn.pack(side='right')
    
    apply_btn = ttk.Button(right_frame, text="Apply", command=result_img_save)
    apply_btn.pack(side='right', padx=15)
    
    reset_after_btn = ttk.Button(right_frame, text="Reset", style='my.TButton', command=reset_after_canvas)
    reset_after_btn.pack(side='left')

        
    #設定對應類別的影像處理方法function名稱
    group_list = {0:fns_negative
                  , 1:fns_histogram_eq
                  , 2:robert_op
                  , 3:prewitt_op
                  , 4:sobel_op
                  , 5:fns_laplacian_op
                  , 6:fns_canny
                  , 7:fns_black_noise
                  , 8:fns_white_noise
                  , 9:morphology_erode
                  , 10:morphology_dilate
                  , 11:morphology_open
                  , 12:morphology_close
                  , 13:morphology_thinning
                  , 14:clahe_l_channel_method
                  , 15:gamma_clahe_l_channel_method
                  , 16:color_normalize_method
                  , 17:clahe_bgr_method
                  , 18:clahe_bgr_g_channel_method
                  , 19:clahe_hsv_v_channel_method
                  , 20:adjust_gamma
                  , 21:circle_crop
                  , 22:gaussianblur
                  , 23:fns_RGB_to_BGR
                  , 24:circle_crop2
                  , 25:fns_medical_img_enhancement}
     
    #存放ttk.Checkbutton的變數字典列表
    tk_var_dict = {}
    
    # 影像處理方法的清單，會顯示在gui介面中間
    fns_name_list = ['Negative'
                     , 'Histogram Equalization'
                     , 'Roberts Edge Detection'
                     , 'Prewitt Edge Detection'
                     , 'Sobel Edge Detection'
                     , 'Laplacian Edge Detection'
                     , 'Canny Edge Detection'
                     , 'Black Noise'
                     , 'White Noise'
                     , 'Morphology Erosion'
                     , 'Morphology Dilation'
                     , 'Morphology Opening'
                     , 'Morphology Closing'
                     , 'Morphology Thinning'
                     , 'CLAHE On LAB(L Channel)'
                     , 'Gamma and CLAHE LAB'
                     , 'Color Normalization'
                     , 'CLAHE On BGR'
                     , 'CLAHE On BGR(G Channel)'
                     , 'CLAHE On HSV(V Channel)'
                     , 'Gamma Correction'
                     , 'Circle Crop'
                     , 'Gaussianblur'
                     , 'RGB To BGR'
                     , 'Circle Crop Method2'
                     , 'Paper Image Enhancement']
    
    # 顯示影像處理方法Checkbutton
    for x, e in enumerate(fns_name_list):
        # 設定每一個Checkbutton變數，存入tk_var_dict
        tk_var_dict[x] = tk.BooleanVar()
        tk_var_dict[x].set(0)
        ttk.Checkbutton(scrollable_algo_frame, text=f'{x+1}.{e}', style='my.TCheckbutton', bootstyle="round-toggle", variable=tk_var_dict[x], command=show_select_list).pack(fill="x")
        


    root.mainloop()

