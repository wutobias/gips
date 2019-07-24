from collections import OrderedDict
import parmed as pmd
import numpy as np
from gips.utils.read_write import loadgist
from gips.utils.read_write import read_gistin
from gips.grid_solvent.spatial import merge_gist
from gips.grid_solvent.spatial import bounding_edge_frac

import copy

###DEBUG
#from gips.utils.read_write import write_maps
#from gips.utils.read_write import write_files

def buildlib(path, verbose=False, cut=-1):

    if verbose:
        print "Parsing gist library file..."
    gdatarec_options, gdata_options = read_gistin(path)
    gdatarec_dict = OrderedDict()
    gdata_dict = OrderedDict()

    allowed_options_gistdata = ["title",
                                "dg",
                                "dh",
                                "ds",
                                "unit",
                                "ref"]

    allowed_options_gistdatarec = ["title"]

    allowed_inits_list = ["gdat",
                        "topo",
                        "strc",
                        "sele",
                        "w"]

    allowed_inits_gistdata = OrderedDict()
    allowed_inits_gistdata["complex"] = allowed_inits_list
    allowed_inits_gistdata["ligand"]  = allowed_inits_list
    allowed_inits_gistdata["pose"   ] = ["topo",
                                        "strc",
                                        "sele",
                                        "ref"]

    allowed_inits_gistdatarec = OrderedDict()
    allowed_inits_gistdatarec["receptor"] = allowed_inits_list

    if verbose:
        i=0
    for options_list, options_dict, allowed_options, allowed_inits in [(gdatarec_options, gdatarec_dict, allowed_options_gistdatarec, allowed_inits_gistdatarec),\
                                                                       (gdata_options,    gdata_dict,    allowed_options_gistdata,    allowed_inits_gistdata)]:

        if verbose:
            i += 1
            if i==1:
                print "Loading receptor data..."
            if i==2:
                print "Loading pose/complex/ligand data..."

        for gist_i, gistdata in enumerate(options_list):
            options_dict[gist_i] = OrderedDict()

            if verbose:
                print "Loading gistdata section %d ..." %gist_i
            for option in allowed_options:
                if option in gistdata.keys():
                    value = gistdata[option]
                    if len(value)>1:
                        raise IOError("Option %s seems to be defined more than once." %option)
                    else:
                        options_dict[gist_i][option] = value[0]
                else:
                    options_dict[gist_i][option] = None

            for init, options in allowed_inits.items():
                if init in gistdata.keys():
                    options_dict[gist_i][init] = list()
                    for initdict_retrieve in gistdata[init]:
                        initdict = OrderedDict()
                        topo     = None
                        strc     = None
                        sele     = None
                        w        = None
                        ref      = None
                        gdatlist = list()
                        for option in options:
                            value = None
                            if option in initdict_retrieve.keys():
                                value = initdict_retrieve[option]
                            else:
                                initdict[option] = None
                                continue
                            if option=="gdat":
                                if value[0]==None:
                                    gdatlist = None
                                elif value==None:
                                    gdatlist = None
                                else:
                                    for gdat in value:
                                        if verbose:
                                            print "Loading %s ..." %gdat
                                        gdatlist.append(loadgist(gdat))
                            elif len(value)>1:
                                raise IOError("Option %s seems to be defined more than once." %option)
                            elif option=="topo":
                                topo = value[0]
                            elif option=="strc":
                                strc = value[0]
                            elif option=="sele":
                                sele = value[0]
                            elif option=="w":
                                w = value[0]
                            elif option=="ref":
                                ref = value[0]
                            else:
                                pass

                        if ref==None and init=="pose":
                            raise IOError("An init pose must be linked to a reference entry.")
                        else:
                            initdict["ref"] = ref

                        ### Topology
                        if topo != None \
                        and strc != None:
                            if verbose:
                                print "Loading %s ..." %topo
                            initdict["pmd"]  = pmd.load_file(topo)
                            if verbose:
                                print "Loading %s ..." %strc
                            initdict["pmd"].load_rst7(strc)
                            if sele != None:
                                if verbose:
                                    print "Making selection %s ..." %sele
                                initdict["pmd"]  = initdict["pmd"][sele]
                        
                        ### Gist data
                        if gdatlist != None:
                            if len(gdatlist)>1:
                                if verbose:
                                    print "Merging gistdata ..."
                                initdict["gdat"] = merge_gist(gdatlist)
                            elif len(gdatlist)==1:
                                initdict["gdat"] = copy.copy(gdatlist[0])
                            else:
                                initdict["gdat"] = None
                        else:
                            initdict["gdat"] = None

                        ### weight factor w
                        if initdict["gdat"] != None:
                            try:
                                w = float(w)
                            except:
                                raise IOError("Option w must be floating point number.")

                        initdict["w"] = w

                        ### Cut the gdata
                        if cut>=0 and initdict["gdat"] != None:

                            if verbose:
                                print "Cutting down grids ..."
                            if not "pmd" in initdict.keys():
                                raise IOError("If using cut, parameter and coordinate files must be provided.")

                            radius_list = list()
                            crds_list   = list()
                            j = 0
                            for a in initdict["pmd"]:
                                if not ( a.name.startswith("h") \
                                or a.name.startswith("H") ):
                                    radius_list.append(a.rmin)
                                    crds_list.append(initdict["pmd"].coordinates[j])
                                j += 1

                            radius_list = np.array(radius_list)+cut
#                            ### First approach:
#                            ### Estimate bounding box by creating a bounding surface
#                            sasa_crds   = sasa_vol(np.array(crds_list,
#                                                initdict["gdat"], 
#                                                radius_list, 
#                                                1.4,
#                                                verbose=verbose)
#                            sasa_max  = np.max(sasa_crds, axis=0)+1
#                            sasa_min  = np.min(sasa_crds, axis=0)-1
#                            cut_bins  = np.zeros((3,2), dtype=np.int)
#                            cut_bins[:,0] = sasa_min.astype(np.int)
#                            cut_bins[:,1] = initdict["gdat"].bins-sasa_max
#
#                            new_center_frac = sasa_min+(sasa_max - sasa_min)/2
#                            initdict["gdat"].center = initdict["gdat"].get_real(new_center_frac)
#                            initdict["gdat"].cut(cut_bins)
#
#                            del sasa_crds

                            ### Second approach:
                            ### Estimate bounding box by increasing the structure coordinates to
                            ### a maximum and minimum structure according to atom radii
                            real_structure = np.array(crds_list)
                            max_atoms      = np.argmax(real_structure, axis=0)
                            min_atoms      = np.argmin(real_structure, axis=0)
                            real_structure[max_atoms[0],0] += radius_list[max_atoms[0]]
                            real_structure[max_atoms[1],1] += radius_list[max_atoms[1]]
                            real_structure[max_atoms[2],2] += radius_list[max_atoms[2]]
                            real_structure[min_atoms[0],0] -= radius_list[min_atoms[0]]
                            real_structure[min_atoms[1],1] -= radius_list[min_atoms[1]]
                            real_structure[min_atoms[2],2] -= radius_list[min_atoms[2]]
                            frac_structure = initdict["gdat"].get_frac(real_structure)
                            bounding_min, bounding_max = bounding_edge_frac(frac_structure)

                            cut_bins      = np.zeros((3,2), dtype=np.int)
                            cut_bins[:,0] = bounding_min.astype(np.int)
                            cut_bins[:,1] = initdict["gdat"].bins-bounding_max

                            new_center_frac = bounding_min+(bounding_max - bounding_min)/2

                            initdict["gdat"].center = initdict["gdat"].get_real(new_center_frac)
                            initdict["gdat"].cut(cut_bins)
                            ###DEBUG
                            #write_files(XYZ=initdict["pmd"]["!(@H=|@h=)"].coordinates, Format='PDB', Filename='testcrds.pdb')
                            #write_maps(initdict["gdat"])

                        options_dict[gist_i][init].append(initdict)

                        del gdatlist

                else:
                    options_dict[gist_i][init] = None

    return gdatarec_dict, gdata_dict