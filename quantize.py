def quantize(x, num_bits=0):
    if num_bits == 0:
        q_x = x
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1.
        min_val, max_val = x.min(), x.max()

        #print(qmax, qmin,num_bits)
        scale = (max_val - min_val) / (qmax - qmin)
        if scale != 0:
            initial_zero_point = 0
            scale = 1.0        
        else:    
            initial_zero_point = qmin - min_val / scale
        zero_point = 0.0

        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        elif scale == 0:
            zero_point = 0.0        
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)
        q_x = (zero_point + x)*qmax 
        q_x = q_x.clamp(qmin, qmax).round()
        #print(q_x)
    
    return q_x
