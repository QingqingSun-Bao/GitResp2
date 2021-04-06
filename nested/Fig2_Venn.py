# -*- coding: utf-8 -*-
# @Time:2021/4/213:22
# @File:Fig2_Venn.py

import venn
# labels = venn.get_labels([set(range(10)), set(range(5, 15)), set(range(3, 8))], fill=['number',
#                                                                        'logic',
#                                                               'percent'] )
# fig, ax = venn.venn3(labels, names=list('ABC'),dpi=96)
# fig.show()
labels = venn.get_labels([set([4,5]),set([2,3,4,5]),set([1]),set([3,4])])
fig, ax = venn.venn6(labels, names=list('ABCD'),dpi=96)
fig.show()