import os  
import shutil
# os.rename('guru99.txt','career.guru99.txt')



directories = []

for dir in os.listdir('./'):
    if os.path.isdir(dir):
        directories.append(f'./{dir}')

directories = sorted(directories)
print(directories)


# i = 60
# for dir in directories:
#     print(dir)
#     for f in os.listdir(dir):
#         t = f.replace('xlsx', 'csv')

#         if 'Accelerometer' in f:
#             os.rename(f'./{dir}/{f}',f'./{dir}/overhead_accel_{i}.csv')

#         if 'Gyroscope' in f:
#             os.rename(f'./{dir}/{f}',f'./{dir}/overhead_gyro_{i}.csv')

#     i += 1

# for dir in directories:
#     # print(dir)
#     for f in os.listdir(dir):
#         shutil.copy(f'./{dir}/{f}', f'./')