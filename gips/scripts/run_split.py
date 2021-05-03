from gips.utils.misc import generate_ksplits

def split(pairfile=None, inclfile=None, exclfile=None, K=5, cut=0.75, prefix=""):

    if pairfile == None and inclfile == None:
        raise ValueError("Must provide pairfile or include file or both.")

    if pairfile=="" and inclfile=="":
        raise ValueError("Must provide pairfile or include file or both.")

    if K<=0:
        raise ValueError("K must be >0.")

    if cut<0.:
        raise ValueError("cut must be >0.")

    exclude_mols=list()
    if exclfile != "" and exclfile != None:
        with open(exclfile, "r") as fopen:
            for line in fopen:
                l = line.lstrip().rstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith('#'):
                    continue
                for s in l:
                    exclude_mols.append(s)

    include_mols=list()
    if inclfile != "" and inclfile != None:
        with open(inclfile, "r") as fopen:
            for line in fopen:
                l = line.lstrip().rstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith('#'):
                    continue
                for s in l:
                    if s in exclude_mols:
                        continue
                    include_mols.append(s)

        L      = len(include_mols)
        splits = generate_ksplits(K, L)
        with open("%ssplits.dat" %prefix, "w") as fopen:
            fopen.write("### Generated for K=%d splits.\n" %K)
            for i in range(L):
                fopen.write(include_mols[i])
                fopen.write(" ")
                fopen.write("%d\n" %splits[i])

    pair_list = list()
    pair_vals = list()
    if pairfile != "" and pairfile != None:
        with open(pairfile, "r") as fopen:
            for line in fopen:
                l = line.lstrip().rstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith('#'):
                    continue
                if l[0] in exclude_mols:
                    continue
                if l[1] in exclude_mols:
                    continue
                c = float(l[2])
                if c>cut:
                    pair_list.append([l[0], l[1]])
                    pair_vals.append(float(l[2]))

        L      = len(pair_list)
        splits = generate_ksplits(K, L)
        with open("%spair-splits.dat" %prefix, "w") as fopen:
            fopen.write("### Generated for K=%d splits and cut=%s.\n" %(K, cut))
            for i in range(L):
                fopen.write(pair_list[i][0])
                fopen.write(" ")
                fopen.write(pair_list[i][1])
                fopen.write(" ")
                fopen.write("%d\n" %splits[i])
