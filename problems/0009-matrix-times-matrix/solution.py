def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	
    m,n = len(a), len(a[0])
    q,p = len(b), len(b[0])

    if n != q: 
        return -1
    
    c = [[0] * p for _ in range(m)]

    for i in range(m): 
        for j in range(p): 
            s = 0
            for k in range(q): 
                s += a[i][k] * b[k][j]
            c[i][j] = s
    
    return c
            


