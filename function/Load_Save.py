# -*- coding: utf-8 -*-
# @Time:2021/3/2415:19
# @File:Load_Save.py
"""导入与保存文件"""

'''导入矩阵字典'''

def LoadDict(path):
    fr = open(path, encoding='utf-8')
    dic = eval(fr.read())  # 将str转化成dict
    fr.close()
    return dic


'''保存文件'''


def Savedict(path,datadict):
    file = open(path, "w",encoding='utf8')
    file.write(str(datadict))
    file.close()


if __name__=="__main__":
    # 主程序的入口
    print("成功导入并保存")