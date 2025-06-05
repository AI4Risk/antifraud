import os
import matplotlib.pyplot as plt

def color_print(content):
    print(f'\033[1;46m{content}\033[0m\n')

def save_pic_iterly(pic_name, postfix, info):
    pic_idx=1
    pic_name_full=f'{pic_name}_{pic_idx}.{postfix}'

    while os.path.exists(pic_name_full):
        print(f'File {pic_name_full} already exists.')
        pic_idx += 1
        pic_name_full=f'{pic_name}_{pic_idx}.png'

    plt.savefig(pic_name_full, dpi=300, bbox_inches='tight')

    color_print(f'!!!!! {info} is saved in file {pic_name_full}')