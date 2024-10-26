import numpy as np
# def fft2(x, dim=(-2,-1)):
#     return np.fft.fft2(x, axes=dim, norm='ortho')
#
# def ifft2(X, dim=(-2,-1)):
#     return np.fft.ifft2(X, axes=dim, norm='ortho')
#
# def fft2c(x, dim=(-2,-1)):
#     return np.fft.fftshift(fft2(np.fft.ifftshift(x, axes=dim), dim), dim)
#
# def ifft2c(x, dim=(-2,-1)):
#     return np.fft.fftshift(ifft2(np.fft.ifftshift(x, axes=dim), dim), dim)
#
#
# # 生成随机的0和1的numpy数组
# random_array = np.random.choice([0, 1], size=(4, 4))
# print("Random array:")
# print(random_array)
#
# # 应用ifft2c函数
# result_fft = fft2c(random_array)
# print("\nResult after ifft2c:")
# print(result_fft)
#
# # 应用fft2c函数
# result_ifft = ifft2c(result_fft)
# print("\nResult after fft2c:")
# print(result_ifft)
#
# # 检查结果是否与原始数组相同
# print("\nIs original array equal to the result of fft2c(ifft2c(random_array))?")
# print(np.array_equal(random_array, result_ifft))




