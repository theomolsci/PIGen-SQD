import numpy as np


# R_eq = 1\AA
# 'arr' defines the stretch factors, so R = x * R_eq where x are the elements of 'arr'
arr = np.array([0.8,0.9,1.0, 1.1,1.2,1.4,1.5,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])#,2.25,2.5,2.75,3.0])


#print (arr)

file4 = "run_array_low.txt"
f4= open(file4, "w")
for i in arr:
    print (i, file=f4)
f4.close
