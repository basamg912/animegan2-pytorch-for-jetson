import cv2
import os
import numpy as np
from pathlib import Path

input_dir = Path("./inputs/")
output_dir = Path("./outputs/")

compare_dir = Path("./compare/")

for i in range(1,4,1):
    input_path = f"{input_dir}/{i}.jpg"
    output_path = f"{output_dir}/{i}.png"
    
    input_image = cv2.imread(input_path)
    input_image = cv2.resize(input_image, (512,512))
    output_image = cv2.imread(output_path)
    
    concat_img = cv2.hconcat([input_image, output_image])
    cv2.imwrite(f"{compare_dir}/{i}.png", concat_img)