import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理器

# ---------------MRR------------------------------
x_data = ['100','200','300','500','1000','1500','2000']

# y_data = [0.541,0.577,0.593,0.607,0.617,0.622,0.623] # TComplex
# y_data2 = [0.549,0.579,0.594,0.604,0.610,0.612,0.614]  # TNTComplex
# y_data3 = [0.536,0.567,0.578,0.592,0.601,0.605,0.604]  # DEComplex
# y_data4 = [0.551,0.589,0.604,0.618,0.627,0.629,0.633]  # joint-Complex
#
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# ln1, = plt.plot(x_data,y_data,color='red',linewidth=2.0,linestyle='--', marker='o', markersize=9)
# ln2, = plt.plot(x_data,y_data2,color='blue',linewidth=2.0,linestyle='--', marker='s', markersize=9)
# ln3, = plt.plot(x_data,y_data3,color='green',linewidth=2.0,linestyle='--', marker='*', markersize=9)
# ln4, = plt.plot(x_data,y_data4,color='black',linewidth=2.0,linestyle='--', marker='X', markersize=9)
#
# my_font = fm.FontProperties()
#
# font2 = {#'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 14,
# }
#
# font3 = {#'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 12,
# }
#
# # plt.title("The comparision of MRR result",fontproperties=my_font)  # 设置标题及字体
# plt.legend(handles=[ln4,ln1,ln2, ln3],labels=['Joint-ComplEx','TComplEx','TNTComplEx','DE-TComplEx'],prop=font3)
# plt.xlabel('Dimension', font2)
# plt.ylabel('MRR', font2)
#
# # 设置数字标签
# # for a, b in zip(x_data, y_data3):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
# #
# # for a, b in zip(x_data, y_data2):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
# #
# # for a, b in zip(x_data, y_data):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
#
# ax = plt.gca()
#
# ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
# ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
#
# plt.savefig('./MRR.png')
# plt.show()

# ---------------Hit1------------------------------
# y_data = [0.449,0.492,0.510,0.525,0.535,0.542,0.542] # TComplex
# y_data2 = [0.453,0.487,0.504,0.516,0.524,0.525,0.527]  # TNTComplex
# y_data3 = [0.443,0.477,0.488,0.505,0.514,0.518,0.517]  # DEComplex
# y_data4 = [0.454,0.499,0.518,0.534,0.546,0.549,0.554]  # joint-Complex
#
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
#
# ln1, = plt.plot(x_data,y_data,color='red',linewidth=2.0,linestyle='--', marker='o', markersize=9)
# ln2, = plt.plot(x_data,y_data2,color='blue',linewidth=2.0,linestyle='--', marker='s', markersize=9)
# ln3, = plt.plot(x_data,y_data3,color='green',linewidth=2.0,linestyle='--', marker='*', markersize=9)
# ln4, = plt.plot(x_data,y_data4,color='black',linewidth=2.0,linestyle='--', marker='X', markersize=9)
#
# my_font = fm.FontProperties()
# font2 = {#'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 14,
# }
#
# font3 = {#'family' : 'Times New Roman',
#          'weight' : 'normal',
#          'size'   : 12,
# }
# # plt.title("The comparision of MRR result",fontproperties=my_font)  # 设置标题及字体
# plt.legend(handles=[ln4,ln1,ln2, ln3],labels=['Joint-ComplEx','TComplEx','TNTComplEx','DE-TComplEx'],prop=font3)
# plt.xlabel('Dimension', font2)
# plt.ylabel('Hit@1', font2)
#
# # 设置数字标签
# # for a, b in zip(x_data, y_data3):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
# # for a, b in zip(x_data, y_data2):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
# #
# # for a, b in zip(x_data, y_data):
# #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
#
# ax = plt.gca()
#
# ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
# ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
#
# plt.savefig('./Hit1.png')
# plt.show()


# ---------------Hit10------------------------------
y_data = [0.712,0.735,0.747,0.757,0.767,0.770,0.772] # TComplex
y_data2 = [0.728,0.749,0.759,0.768,0.770,0.774,0.777]  # TNTComplex
y_data3 = [0.712,0.737,0.744,0.753,0.760,0.764,0.763]  # DEComplex
y_data4 = [0.727,0.755,0.762,0.772,0.775,0.781,0.782]  # joint-Complex

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ln1, = plt.plot(x_data,y_data,color='red',linewidth=2.0,linestyle='--', marker='o', markersize=9)
ln2, = plt.plot(x_data,y_data2,color='blue',linewidth=2.0,linestyle='--', marker='s', markersize=9)
ln3, = plt.plot(x_data,y_data3,color='green',linewidth=2.0,linestyle='--', marker='*', markersize=9)
ln4, = plt.plot(x_data,y_data4,color='black',linewidth=2.0,linestyle='--', marker='X', markersize=9)

font2 = {#'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 14,
}

font3 = {#'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 12,
}
# plt.title("The comparision of MRR result",fontproperties=my_font)  # 设置标题及字体
plt.legend(handles=[ln4,ln1,ln2, ln3],labels=['Joint-ComplEx','TComplEx','TNTComplEx','DE-TComplEx'],prop=font3)
plt.xlabel('Dimension', font2)
plt.ylabel('Hit@10', font2)

# 设置数字标签
# for a, b in zip(x_data, y_data3):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
# for a, b in zip(x_data, y_data2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
#
# for a, b in zip(x_data, y_data):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
ax = plt.gca()

ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示

plt.savefig('./Hit10.png')
plt.show()