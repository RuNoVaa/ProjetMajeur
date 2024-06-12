def triangle(pt1, pt2, pt3,k):

    k=k//3 # Divsion par 3 pour avoir K points au total sur nos 3 côtés
    pas12=(pt2[0]-pt1[0])/k
    pas23=(pt3[0]-pt2[0])/k
    pas31=(pt1[0]-pt3[0])/k

    #Listes des coefficients directeurs de chaque côté
    coeff_dir12=(pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    coeff_dir23=(pt3[1]-pt2[1])/(pt3[0]-pt2[0])
    coeff_dir31=(pt1[1]-pt3[1])/(pt1[0]-pt3[0])

    #ordonnée à l'origine 
    b12=pt1[1]-coeff_dir12*pt1[0]
    b23=pt2[1]-coeff_dir23*pt2[0]
    b31=pt3[1]-coeff_dir31*pt3[0]

    #Liste points des 3 segments
    point12=[pt1]
    point23=[pt2]
    point31=[pt3]

    #On trace les segments
    for i in range(k-1):
        x12=point12[i][0]+pas12
        x23=point23[i][0]+pas23
        x31=point31[i][0]+pas31
        point12.append((x12,x12*coeff_dir12+b12))
        point23.append((x23,x23*coeff_dir23+b23))
        point31.append((x31,x31*coeff_dir31+b31))
    
    #Concaténation des listes de points
    points=point12+point23+point31
    x = [x for x, y in points]
    y = [y for x, y in points]
    return x, y