#coding=utf-8
#实现多项式采样
import numpy
def multinomial_sample(pro_list):
        test_list = []
        test_list.append(pro_list[0])
        for k in range(1, len(pro_list)):
            #pro_list[k] += pro_list[k-1]
            test_list.append(pro_list[k] + test_list[k-1])
            
        u = numpy.random.rand() * test_list[-1]
        #print(test_list)
        return_index = len(test_list) - 1
        for t in range(len(test_list)):
            if test_list[t] > u:
                return_index = t
                break
        return return_index
    
if __name__=='__main__':
    pro_list = [0.2, 0.3, 0.4, 0.1]
    list = []
    print(multinomial_sample(pro_list))
    i = 0
    while i<1000:
        list.append(multinomial_sample(pro_list))
        #print(multinomial_sample(pro_list))
        i = i+1
    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    for t in range(len(list)):
        if list[t] == 0:
            i0=i0+1
        elif list[t] == 1:
            i1=i1+1
        elif list[t] == 2:
            i2=i2+1
        elif list[t] == 3:
            i3=i3+1
    print(i0/len(list),i1/len(list),i2/len(list),i3/len(list))
    print(i0,i1,i2,i3,len(list))
