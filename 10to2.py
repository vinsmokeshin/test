#!/usr/bin/env python
s=input("s:")
from string import *
a=abs(int(s))
b=int(a)
c=[]
e=[]
a=a-b
while int(b/2)>0:
    c=c+[b%2]
    b=int(b/2)
c=c+[1]
c.reverse()
for i in range(len(c)):
    c[i]=str(c[i])
c=join(c)
while a*2!=0 or len(e)<=15:
    e=e+[(int(a*2))]
    a=a*2
    if a>=1:
        a=a-1
for i in range(len(e)):
    e[i]=str(e[i])
e=join(e)
if s<0:
    print("-",c,".",e)
if s>=0:   
    print(c,".",e)



    
