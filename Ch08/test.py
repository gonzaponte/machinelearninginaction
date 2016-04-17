from regression import loadDataSet
from Multivariate import LinearRegresion
from ROOT import *
from array import array
from math import *


xs, ys = loadDataSet('ex0.txt')
x0, x1 = zip(*xs)

ndata = len(ys)
nregr = 100
dregr = 1.0/nregr

lr   = LinearRegresion(xs,ys)
lwlr = LinearRegresion(xs,ys,lambda x,y: exp(-(x-y)**(x-y)/(2*0.01**2)), 0.0001 )

xlr = [ (1.0,dregr*i) for i in range(nregr) ]
ylr = map( lr.GetValue, xlr )

xlwlr = xlr
ylwlr = map( lwlr.GetValue, xlwlr )

gdata = TGraph( ndata, array('f',x1)   , array('f',ys) )
glr   = TGraph( nregr, array('f',zip(*xlr)[1])  , array('f',ylr) )
glwlr = TGraph( nregr, array('f',zip(*xlwlr)[1]), array('f',ylwlr) )

gdata.SetMarkerStyle(20)
glr  .SetLineWidth(2)
glwlr.SetLineWidth(2)
glr  .SetLineColor(kRed)
glwlr.SetLineColor(kBlue)

gdata.Draw('ap')
glr  .Draw('csame')
glwlr.Draw('csame')

raw_input('done')
