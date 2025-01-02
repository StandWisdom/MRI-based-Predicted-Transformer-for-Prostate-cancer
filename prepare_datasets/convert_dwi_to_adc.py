# 导入需要的模块
import nibabel as nib
import numpy as np
import os

# 读取高B值和低B值的nii文件
srcDir = r'G:\data\Prostate(in-house)\PCaDeepSet\NewTest\CZ01'
high_b = nib.load(os.path.join(srcDir,'b800.nii.gz'))
low_b = nib.load(os.path.join(srcDir,'b50.nii.gz'))

# 获取图像数据和仿射矩阵
high_b_data = high_b.get_fdata()
low_b_data = low_b.get_fdata()
affine = high_b.affine

# 计算ADC序列
b1 = 50 # 低B值
b2 = 800 # 高B值
adc_data = np.log(low_b_data / high_b_data) / (b2 - b1)

# 创建ADC图像并保存为nii文件
adc = nib.Nifti1Image(adc_data, affine)
nib.save(adc, os.path.join(srcDir,'adc.nii.gz'))