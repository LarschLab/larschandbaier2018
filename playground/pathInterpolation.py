# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:33:41 2017

@author: jlarsch
"""
import numpy as np
import math
import matplotlib.pyplot as plt

part=np.arange(0,10*60*30,1)
x=experiment.pair.animals[0].ts.rawTra().x()[108000:108000+10*60*30]
y=experiment.pair.animals[0].ts.rawTra().y()[108000:108000+10*60*30]
#xd=np.diff(x)
#yd=np.diff(y)
#dist = np.sqrt(xd**2+yd**2)
#u = np.cumsum(dist)
#u = np.hstack([[0],u])
#
#t=np.linspace(0,u.max(),u.shape[0])
#xn = np.interp(t, u, x)
#yn = np.interp(t, u, y)
#fig, ax = plt.subplots()
#ax.plot(x,y,'-.')
#ax.plot(xn,yn,'go-')
#ax.set_aspect('equal')
#plt.show()




M = len(x)*100
t = np.linspace(0, len(x), M)
xi = np.interp(t, np.arange(len(x)), x)
yi = np.interp(t, np.arange(len(y)), y)


tol = 2
i, idx = 0, [0]
while i < len(xi)-1:
    total_dist = 0
    for j in range(i+1, len(xi)):
        total_dist = math.sqrt((xi[j]-xi[i])**2 + (yi[j]-yi[i])**2)
        if total_dist > tol:
            idx.append(j)
            break
    i = j+1

xn = xi[idx]
yn = yi[idx]

# Interpolate values for x and y.
t = np.arange(len(xn))
t2 = np.linspace(0, len(xn), len(x))
# One-dimensional linear interpolation.
xnn = np.interp(t2, t, xn)
ynn = np.interp(t2, t, yn)

fig, ax = plt.subplots()
ax.plot(xnn[part],ynn[part],'go-')
#ax.plot(xi[0:M/2], yi[0:M/2], '.-')
#ax.plot(x[part], y[part], 'sr')
ax.set_aspect('equal')
plt.show()

np.savetxt(avi_path+".csv", np.array([xnn,ynn]).T, '%10.5f',delimiter=",")

t=np.arange(0,10*60*30*30,30)/1000.0
xs=np.sin(t*(1.5/5.4))*180+256
ys=np.cos(t*(1.5/5.4))*180+256
plt.figure()
plt.plot(xs,ys)


xnnd=np.diff(xnn)
ynnd=np.diff(ynn)
nndist = np.sqrt(xnnd**2+ynnd**2)
plt.figure()
plt.hist(nndist,100)

xd=np.diff(x)
yd=np.diff(y)
xd=np.hstack([0,xd])
yd=np.hstack([0,yd])
dist = np.sqrt(xd**2+yd**2)



plt.figure()
plt.hist(dist,100)

xsd=np.diff(xs)
ysd=np.diff(ys)
xsd=np.hstack([0,xsd])
ysd=np.hstack([0,ysd])
sdist = np.sqrt(xsd**2+ysd**2)
plt.figure()
plt.hist(sdist,100)

tscale=30*(dist*(1/5.4))/1000.0
tscaleCum=np.cumsum(tscale)

xs2=np.sin(tscaleCum)*180+256
ys2=np.cos(tscaleCum)*180+256
plt.figure()
plt.plot(xs2,ys2)
xsd2=np.diff(xs2)
ysd2=np.diff(ys2)
xsd2=np.hstack([0,xsd2])
ysd2=np.hstack([0,ysd2])
sdist2 = np.sqrt(xsd2**2+ysd2**2)
plt.figure()
plt.hist(sdist2,100)


modeList1=np.repeat('skype',len(t))
modeList2=np.repeat('fix_real_rSpeed',len(t))
modeList3=np.repeat('fix_real_cSpeed',len(t))
modeList4=np.repeat('fix_circ_cSpeed',len(t))
modeList5=np.repeat('fix_circ_rSpeed',len(t))

modeListAll=np.hstack([modeList1,modeList2,modeList3,modeList4,modeList5])
xskype=np.repeat(0,len(t))
yskype=np.repeat(0,len(t))



xAll=np.hstack([xskype,x,xnn,xs,xs2])
yAll=np.hstack([yskype,y,ynn,ys,ys2])

ab = np.zeros(xAll.size, dtype=[('var1', 'S32'), ('var2', float),('var3', float)])
ab['var1'] = modeListAll
ab['var2'] = xAll
ab['var3'] = yAll

np.savetxt(avi_path+".csv", ab, fmt="%s, %10.3f, %10.3f")

#np.savetxt(avi_path+".csv", np.array([modeListAll,xAll,xAll]).T, ['%s','%10.5f','%10.5f'],delimiter=",")
