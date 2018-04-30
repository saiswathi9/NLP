
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import xlsxwriter

bot = pd.read_excel('C:/Users/sivar/Desktop/sai/nlp.xlsx').as_matrix()
nonbot = pd.read_excel('C:/Users/sivar/Desktop/sai/nlp1.xlsx').as_matrix()
X=np.concatenate((bot,nonbot))
print(X)
workbook = xlsxwriter.Workbook('C:/Users/sivar/Desktop/res_bin.xlsx')
worksheet = workbook.add_worksheet()

array=X
row = 0
col=0

for wdr,dissim,leven in (array):
    worksheet.write(row,col, wdr)
    worksheet.write(row,col+1, dissim) 
    worksheet.write(row,col+2, leven)
   
    row+=1
workbook.close()


# In[59]:


import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/sivar/Desktop/res1.csv', sep=',')['wdr'].hist(bins=20)
plt.xlabel('wdr')
plt.ylabel('frequency')
plt.title('wdr histogram')


#hist,bins=np.histogram(X,bins=20)
#print(N)
#width = 0.9 * (bins[1] - bins[0])
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist, align='center', width=width)
#plt.show()


# In[90]:


import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('C:/Users/sivar/Desktop/res1.csv', sep=',',quoting=1)['dissim'].hist(bins=20)
plt.xlabel('dissim')
plt.ylabel('frequency')
plt.title('dissim plot')


# In[61]:


import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('C:/Users/sivar/Desktop/res1.csv', sep=',',quoting=2)['leven'].hist(bins=20)
plt.xlabel('leven')
plt.ylabel('frequency')
plt.title('leven plot')


# In[83]:


import numpy as np
import statsmodels.api as sm
import pylab
print('wdr')
measurements = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
sm.qqplot(measurements, fit=True,line='45')
pylab.show()


# In[84]:


import numpy as np
import statsmodels.api as sm
import pylab
print('dissim')
measurements = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
sm.qqplot(measurements, fit=True,line='45')
pylab.show()


# In[87]:


import numpy as np
import statsmodels.api as sm
import pylab
print('leven')
measurements = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
sm.qqplot(measurements, fit=True,line='45')
pylab.show()


# In[103]:


import numpy
import sys
df = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
p=df.as_matrix()
bins = numpy.linspace(-0.02,0.12,20)
print(bins)
pos = numpy.digitize(p, bins)
for n in range(len(p)):
    sys.stdout.write("%g <= %g < %g\n"
        %(bins[pos[n]-1], p[n], bins[pos[n]]))


# In[143]:


import numpy
import sys
import csv
df = pd.read_excel('C:/Users/sivar/Desktop/res_bin.xlsx')
n=df.as_matrix()
p=n[:, [0]]
q=n[:, [1]]
      
for i in range(len(p)):
    if(p[i]>=-0.02 and p[i]<=-0.01263158):
        p[i]=1
        print(p[i],"\n",q[i],end="")
    if(p[i]>=-0.01263158 and p[i]<=-0.00526316):
        p[i]=2
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=-0.00526316 and p[i]<=0.00210526):
        p[i]=3
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.00210526 and p[i]<=0.00947368):
        p[i]=4
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.00947368 and p[i]<=0.01684211):
        p[i]=5
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.01684211 and p[i]<=0.02421053):
        p[i]=6
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.02421053 and p[i]<=0.03157895):
        p[i]=7
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.03157895 and p[i]<=0.03894737):
        p[i]=8
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.03894737 and p[i]<=0.04631579):
        p[i]=9
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.04631579 and p[i]<=0.05368421):
        p[i]=10
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.05368421 and p[i]<=0.06105263):
        p[i]=11
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.06105263 and p[i]<=0.06842105):
        p[i]=12
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.06842105 and p[i]<=0.07578947):
        p[i]=13
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.07578947 and p[i]<=0.08315789):
        p[i]=14
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.08315789 and p[i]<=0.09052632):
        p[i]=15
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.09052632 and p[i]<=0.09789474):
        p[i]=16
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.09789474 and p[i]<=0.10526316):
        p[i]=17
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.10526316 and p[i]<=0.11263158):
        p[i]=18
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.11263158 and p[i]<= 0.12):
        p[i]=19
        print(p[i],"\n",q[i],end="")
csvfile = "C:/Users/sivar/Desktop/test.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(p)
    
csvfile = "C:/Users/sivar/Desktop/test1.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(q)


# In[105]:


import numpy
import sys
df = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
p=df.as_matrix()
bins = numpy.linspace(0,1,20)
print(bins)
#pos = numpy.digitize(p, bins)
#for n in range(len(p)):
#    sys.stdout.write("%g <= %g < %g\n"
#        %(bins[pos[n]-1], p[n], bins[pos[n]]))


# In[141]:


import numpy
import sys
import csv
df = pd.read_excel('C:/Users/sivar/Desktop/res_bin.xlsx')
n=df.as_matrix()
p=n[:, [0]]
q=n[:, [1]]
       
for i in range(len(p)):
    if(p[i]>=0 and p[i]<=0.05263158):
        p[i]=1
        print(p[i],"\n",q[i],end="")
    if(p[i]>=0.05263158 and p[i]<=0.10526316):
        p[i]=2
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.10526316 and p[i]<=0.15789474):
        p[i]=3
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.15789474 and p[i]<=0.21052632):
        p[i]=4
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.21052632 and p[i]<=0.26315789):
        p[i]=5
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.26315789 and p[i]<=0.31578947):
        p[i]=6
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.31578947 and p[i]<=0.36842105):
        p[i]=7
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.36842105 and p[i]<=0.42105263):
        p[i]=8
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.42105263 and p[i]<=0.47368421):
        p[i]=9
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.47368421 and p[i]<=0.52631579):
        p[i]=10
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.52631579 and p[i]<=0.57894737):
        p[i]=11
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.57894737 and p[i]<=0.63157895):
        p[i]=12
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.63157895 and p[i]<=0.68421053):
        p[i]=13
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.68421053 and p[i]<=0.73684211):
        p[i]=14
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.73684211 and p[i]<=0.78947368):
        p[i]=15
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.78947368 and p[i]<=0.84210526):
        p[i]=16
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.84210526 and p[i]<=0.89473684):
        p[i]=17
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.89473684 and p[i]<=0.94736842):
        p[i]=18
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=0.94736842 and p[i]<= 1):
        p[i]=19
        print(p[i],"\n",q[i],end="")
csvfile = "C:/Users/sivar/Desktop/test.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(p)
    
csvfile = "C:/Users/sivar/Desktop/test1.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(q) 


# In[107]:


import numpy
import sys
df = pd.read_excel('C:/Users/sivar/Desktop/res.xlsx')
p=df.as_matrix()
bins = numpy.linspace(0,2200,20)
print(bins)


# In[139]:


import numpy
import sys
import csv
df = pd.read_excel('C:/Users/sivar/Desktop/res_bin.xlsx')
n=df.as_matrix()
p=n[:, [0]]
q=n[:, [1]]
#print(q)
for i in range(len(p)):
    if(p[i]>=0 and p[i]<=115.78947368):
        p[i]=1
        print(p[i],"\n",q[i],end="")
    if(p[i]>=115.78947368 and p[i]<=231.57894737):
        p[i]=2
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=231.57894737 and p[i]<=347.36842105):
        p[i]=3
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=347.36842105 and p[i]<=463.15789474):
        p[i]=4
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=463.15789474 and p[i]<=578.94736842):
        p[i]=5
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=578.94736842 and p[i]<=694.73684211):
        p[i]=6
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=694.73684211 and p[i]<=810.52631579):
        p[i]=7
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=810.52631579 and p[i]<=926.31578947):
        p[i]=8
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=926.31578947 and p[i]<=1042.10526316):
        p[i]=9
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1042.10526316 and p[i]<=1157.89473684):
        p[i]=10
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1157.89473684 and p[i]<=1273.68421053):
        p[i]=11
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1273.68421053 and p[i]<=1389.47368421):
        p[i]=12
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1389.47368421 and p[i]<=1505.26315789):
        p[i]=13
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1505.26315789 and p[i]<=1621.05263158):
        p[i]=14
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1621.05263158 and p[i]<=1736.84210526):
        p[i]=15
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1736.84210526 and p[i]<=1852.63157895):
        p[i]=16
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1852.63157895 and p[i]<=1968.42105263):
        p[i]=17
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=1968.42105263 and p[i]<=2084.21052632):
        p[i]=18
        print(p[i],"\n",q[i],end="")
    elif(p[i]>=2084.21052632 and p[i]<= 2200):
        p[i]=19
        print(p[i],"\n",q[i],end="")

csvfile = "C:/Users/sivar/Desktop/test.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(p)
csvfile = "C:/Users/sivar/Desktop/test1.csv"

#Assuming res is a list of lists
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(q)    
    


