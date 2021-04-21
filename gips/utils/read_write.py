import numpy as np
from collections import OrderedDict
from string import ascii_uppercase

from gips.utils._read_ext import read_gist_ext
from gips.datastrc.gist import gist
from gips.utils.misc import are_you_numpy

from gips import FLOAT
from gips import DOUBLE

### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com

def read_parmsfile(path):

    parmdict = OrderedDict()

    with open(path, "r") as fopen:

        for line in fopen:
            l = line.rstrip().lstrip().split()
            if len(l)==0:
                continue
            if l[0]=="###":
                if len(l)<2:
                    continue
                elif l[1]=="Step":
                    parmdict["header"] = list()
                    for entry in l[1:]:
                        parmdict["header"].append(entry)
                elif l[1]=="Best":
                    return parmdict
                else:
                    continue
            if l[0].startswith('#'):
                continue
            parmdict[l[0]] = list()
            for entry in l[1:]:
                parmdict[l[0]].append(float(entry))

    return parmdict

def read_pairsfile(path, cutoff):

    __doc__="""
    Read in a pairs file and returns a list of pairs.
    A proper pairs file looks like this:

    MOL1 MOL2 0.45
    MOL1 MOL5 0.87
    MOL2 MOL7 0.21
    ...

    MOL1,MOL2,etc ... are the names/titles of the datapoints as
    present in the input file. The third column is the score of 
    a pair. Only those pairs will be returned, which satisfy the
    condition score>cutoff.
    """
    
    pairlist = list()

    with open(path, "r") as fopen:
        i=-1
        for line in fopen:
            l = line.rstrip().lstrip().split()
            i += 1
            if len(l)==0:
                continue
            if l[0].startswith('#'):
                continue
            if len(l)>3:
                raise IOError("Line %d contains more than three columns." %i)
            if float(l[2])>cutoff:
                pairlist.append([l[0], l[1]])

    return pairlist


def read_boundsfile(path):

    __doc__="""
    Here we read a file that describes the bounds for the parameters
    that shall be used during a gistfit. Example file:

    E       -10     +10
    S       -10     +10
    e_co    -10     +10
    s_co    -10     +10
    g_co     +2     +20
    C       -10     +10

    """

    allowed_parms = ['E',
                     'S',
                     'e_co',
                     's_co',
                     'g_co',
                     'C']

    parmdict      = OrderedDict()
    with open(path, "r") as fopen:
        for line_idx, line in enumerate(fopen):
            l = line.rstrip().lstrip().split()
            if len(l)==0:
                continue
            if l[0].startswith('#'):
                continue
            if l[0] in allowed_parms:
                if len(l)!=3:
                    raise IOError("Parameter must be followed by two float values at line %d." %(line_idx+1))
                parmdict[l[0]] = [float(l[1]), float(l[2])]
            else:
                raise IOError("Parameter %s at line %d not understood." %(l[0], (line_idx+1)))

    return parmdict


def read_gistin(path):

    __doc__="""
    Reads in gistdata input file and returns it as a
    python digestible format (i.e. mainly dictionary-based).
    """

    ###
    ### opts[0]["title"][0]    = "Test 1"
    ### opts[0]["complex"][0]["gist"][0] = "rec0/gist1/gist.dat"
    ### opts[0]["complex"][0]["gist"][1] = "rec0/gist2/gist.dat"
    ### opts[0]["complex"][1]["gist"][0] = "rec1/gist1/gist.dat"
    ### opts[0]["complex"][1]["gist"][1] = "rec1/gist2/gist.dat"
    ### 
    ### The general idea is.
    ### Store all options in a python dict, where the option name/key
    ### is the key in the dict, an the assigned value(s) of the option
    ### are the values stored in a list of the respective key-value
    ### pair. All init options (e.g. "init complex") are also stored as
    ### a key-value pair, however, the value is again a python dictionary,
    ### which again is filled with key-values pair as described above.
    ###
    ### Note: We actually don't use python dicts, but OrderedDict()
    ### from the collections library.
    ###

    def __raiseIO(line_idx):
        raise IOError("Line %d in input file not understood." %line_idx)

    allowed_options_gistdata = ["title",
                                "dg",
                                "dh",
                                "ds",
                                "unit"]

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
    allowed_inits_gistdatarec["receptor"]  = allowed_inits_list

    read_gistdata    = False
    read_gistdatarec = False
    read_init        = False
    name_init        = ""

    gdata_list = list()
    gdatarec_list = list()
    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            l = line.rstrip().lstrip().split()

            ### Empty lines and comments
            if len(l)==0:
                continue

            if l[0].startswith("#"):
                continue

            ### Invalid commands
            if len(l) != 2:
                __raiseIO(line_idx+1)

            if read_init:
                if not read_gistdata \
                and not read_gistdatarec:
                    __raiseIO(line_idx-1)

            if read_gistdata and read_gistdatarec:
                __raiseIO(line_idx-1)

            if l[0]=="init":
                if l[1]=="gistdata":
                    gdata_list.append(OrderedDict())
                    read_gistdata = True

                elif l[1]=="gistdata-rec":
                    gdatarec_list.append(OrderedDict())
                    read_gistdatarec = True

                elif l[1] in list(allowed_inits_gistdata.keys()):
                    name_init = l[1]
                    if name_init not in list(gdata_list[-1].keys()):
                        gdata_list[-1][name_init] = list()
                    gdata_list[-1][name_init].append(OrderedDict())
                    read_init = True

                elif l[1] in list(allowed_inits_gistdatarec.keys()):
                    name_init = l[1]
                    if name_init not in list(gdatarec_list[-1].keys()):
                        gdatarec_list[-1][name_init] = list()
                    gdatarec_list[-1][name_init].append(OrderedDict())
                    read_init = True

                else:
                    __raiseIO(line_idx+1)

            elif l[0]=="end":
                if l[1]=="gistdata":
                    read_gistdata = False

                elif l[1]=="gistdata-rec":
                    read_gistdatarec = False

                elif l[1]==name_init:
                    read_init = False
                    name_init = ""
                else:
                    __raiseIO(line_idx+1)

            elif read_init:
                if read_gistdata and l[0] in allowed_inits_gistdata[name_init]:
                    if l[0] not in list(gdata_list[-1][name_init][-1].keys()):
                        gdata_list[-1][name_init][-1][l[0]] = list()
                    gdata_list[-1][name_init][-1][l[0]].append(l[1])
                elif read_gistdatarec and l[0] in allowed_inits_gistdatarec[name_init]:
                    if l[0] not in list(gdatarec_list[-1][name_init][-1].keys()):
                        gdatarec_list[-1][name_init][-1][l[0]] = list()
                    gdatarec_list[-1][name_init][-1][l[0]].append(l[1])
                else:
                    __raiseIO(line_idx+1)

            elif read_gistdata and l[0] in allowed_options_gistdata:
                if l[0] not in list(gdata_list[-1].keys()):
                    gdata_list[-1][l[0]] = list()
                gdata_list[-1][l[0]].append(l[1])

            elif read_gistdatarec and l[0] in allowed_options_gistdatarec:
                if l[0] not in list(gdatarec_list[-1].keys()):
                    gdatarec_list[-1][l[0]] = list()
                gdatarec_list[-1][l[0]].append(l[1])

            else:
                __raiseIO(line_idx+1)

    return gdatarec_list, gdata_list


def write_maps(gistobject, prefix="gist", pymol=True):

    ref_energy = -11.063656

    data_dict = OrderedDict()

    data_dict["_Pop.dx"]                = [ gistobject.Pop                                      , 1.0 ]
    data_dict["_gO.dx"]                 = [ gistobject.gO                                       , 4.0 ]
    data_dict["_gH.dx"]                 = [ gistobject.gH                                       , 4.0 ]
    data_dict["_dTStrans_dens.dx"]      = [ gistobject.dTStrans_dens                            , 0.2 ]
    data_dict["_dTStrans_norm.dx"]      = [ gistobject.dTStrans_norm                            , 1.0 ]
    data_dict["_dTSorient_dens.dx"]     = [ gistobject.dTSorient_dens                           , 0.2 ]
    data_dict["_dTSorient_norm.dx"]     = [ gistobject.dTSorient_norm                           , 1.0 ]
    data_dict["_dTSsix_dens.dx"]        = [ gistobject.dTSsix_dens                            , 0.2 ]
    data_dict["_dTSsix_norm.dx"]        = [ gistobject.dTSsix_norm                            , 1.0 ]
    data_dict["_Esw_dens.dx"]           = [ gistobject.Esw_dens                                 , 0.2 ]
    data_dict["_Esw_norm.dx"]           = [ gistobject.Esw_norm                                 , 1.0 ]
    data_dict["_Eww_dens.dx"]           = [ gistobject.Eww_dens                                 , 0.2 ]
    data_dict["_Eww_norm_unref.dx"]     = [ gistobject.Eww_norm_unref                           , 1.0 ]
    data_dict["_Eww_norm_ref.dx"]       = [ gistobject.Eww_norm_unref - ref_energy                 , 1.0 ]
    data_dict["_Eww_norm_ref_dens.dx"]  = [(gistobject.Eww_norm_unref - ref_energy)*gistobject.gO*0.0332, 1.0 ]
    data_dict["_Dipole_x_dens.dx"]      = [ gistobject.Dipole_x_dens                            , 1.0 ]
    data_dict["_Dipole_y_dens.dx"]      = [ gistobject.Dipole_y_dens                            , 1.0 ]
    data_dict["_Dipole_z_dens.dx"]      = [ gistobject.Dipole_z_dens                            , 1.0 ]
    data_dict["_Dipole_dens.dx"]        = [ gistobject.Dipole_dens                              , 1.0 ]
    data_dict["_Neighbor_dens.dx"]      = [ gistobject.Neighbor_dens                            , 0.5 ]
    data_dict["_Neighbor_norm.dx"]      = [ gistobject.Neighbor_norm                            , 1.0 ]
    data_dict["_Order_norm.dx"]         = [ gistobject.Order_norm                               , 5.5 ]
    data_dict["_dTS_dens.dx"]           = [ gistobject.dTStrans_dens+gistobject.dTSorient_dens        , 0.2 ]
    data_dict["_dTS_norm.dx"]           = [ gistobject.dTStrans_norm+gistobject.dTSorient_norm        , 1.0 ]
    data_dict["_E_dens.dx"]             = [ gistobject.Esw_dens+gistobject.Eww_dens                   , 0.2 ]
    data_dict["_E_norm.dx"]             = [ gistobject.Esw_norm+2*(gistobject.Eww_norm_unref-ref_energy), 1.0 ]
    ### 5.098076 is the average Neighbor_norm value for TIP4P-Ew water model
    data_dict["_Neighbor_loss_norm.dx"] = [ gistobject.Neighbor_norm - 5.098076                 , 0.5 ]

    if pymol:

        pymol_string = ""
        pymol_string += "from pymol import cmd\n"
        pymol_string += "\n"

    for name, data in list(data_dict.items()):

        write_files(Frac2Real=gistobject.get_nice_frac2real(), 
            Bins=gistobject.bins, 
            Origin=gistobject.origin, 
            Value=data[0], 
            Format="DX", 
            Filename=prefix+name, 
            Nan_fill=-999)

        if pymol:

            new_name      = str(prefix+name).replace(".dx", "")

            pymol_string += "### %s ###\n" %new_name
            pymol_string += "cmd.load(\"./%s\")\n" %(prefix+name)
            pymol_string += "cmd.isomesh(\"%s\", \"%s\", level=%s)\n" %(new_name+"_map", new_name, data[1])
            pymol_string += "cmd.map_double(\"%s\")\n" %(new_name)
            pymol_string += "\n"

    if pymol:

        pymol_string += "cmd.disable(\"*_map\")\n"
        pymol_string += "cmd.do(\"color blue, *_map\")\n"
        pymol_string += "cmd.do(\"set mesh_negative_color, red\")\n"
        pymol_string += "cmd.do(\"set mesh_negative_visible\")\n"

        with open(prefix+"_pymol.py", "w") as f:
            f.write(pymol_string)

class loadgist(gist):

### Some notes:
### ~~~~~~~~~~~
### - all coordinates are internally treated as origin-free, 
###   therefore the coordinate systems' origin is always at (0,0,0)
### - coordinate systems are right-handed
### - global translations affect everything, non-global translations
###   effect grid only
### - ...same holds for rotations.
### - all rotations and translations affect real space only
### - the effect of rotations and translations is only calculated when
###   real space information is retrieved
### - frac2real matrix as well as its inverse, internally remain unchanged,
###   true frac2real (or its inverse) is available via get_nice_frac2real()
### - all rotations are performed counter-clockwise

    def __init__(self, Path, Readpython=False):

        self.path   = Path
        self.gist17 = True

        if Readpython:
            gist_data = self.read_python()
        else:
            gist_data = self.read_C()
        self._init_field(gist_data)
        self._init_data(gist_data)

        del gist_data

    
    def read_C(self):

        if self.path.endswith(".gz"):
            raise IOError("Cannot read gzip files with the read_C routine. Must initiliaze \
loadgist with Readpython=True.")

        gist_data = read_gist_ext(self.path, 0)

        if gist_data.shape[1]==24:
            self.gist17 = True
        else:
            self.gist17 = False

        return gist_data


    def read_python(self):

        gist_data = np.loadtxt(self.path, comments=["#", "GIST", "voxel"])

        if gist_data.shape[1]==24:
            self.gist17 = True
        else:
            self.gist17 = False

        return gist_data


    def _init_field(self, gist_data):

        bins    = np.zeros(3, dtype=int)
        origin  = np.zeros(3)
        delta   = np.zeros(3)
        dim_min = np.zeros(3)
        dim_max = np.zeros(3)

        z_edge  = np.where(gist_data[:,3]==gist_data[0,3])[0]
        bins[2] = z_edge[1]

        y_edge  = np.where(gist_data[z_edge,2]==gist_data[z_edge[0],2])[0]
        bins[1] = y_edge[1]

        bins[0] = y_edge.shape[0]

        dim_min[0], dim_max[0] = np.min(gist_data[:,1]), np.max(gist_data[:,1])
        dim_min[1], dim_max[1] = np.min(gist_data[:,2]), np.max(gist_data[:,2])
        dim_min[2], dim_max[2] = np.min(gist_data[:,3]), np.max(gist_data[:,3])

        delta   = (dim_max-dim_min)/(bins-1)
        origin  = gist_data[0,[1,2,3]]

        super(loadgist, self).__init__(Bins=bins,
                                       Origin=origin,
                                       Delta=delta)

        del bins
        del origin
        del delta
        del dim_min
        del dim_max

        del z_edge
        del y_edge

    def _init_data(self, gist_data):

        offset = 0
        if self.gist17:
            offset = 2

        self.Pop            [:] = gist_data[:,4].reshape(self.bins).astype(FLOAT)
        self.gO             [:] = gist_data[:,5].reshape(self.bins).astype(FLOAT)
        self.gH             [:] = gist_data[:,6].reshape(self.bins).astype(FLOAT)
        self.dTStrans_dens  [:] = gist_data[:,7].reshape(self.bins).astype(FLOAT)
        self.dTStrans_norm  [:] = gist_data[:,8].reshape(self.bins).astype(FLOAT)
        self.dTSorient_dens [:] = gist_data[:,9].reshape(self.bins).astype(FLOAT)
        self.dTSorient_norm [:] = gist_data[:,10].reshape(self.bins).astype(FLOAT)
        if self.gist17:
            self.dTSsix_dens [:]= gist_data[:,11].reshape(self.bins).astype(FLOAT)
            self.dTSsix_norm [:]= gist_data[:,12].reshape(self.bins).astype(FLOAT)
        self.Esw_dens       [:] = gist_data[:,11+offset].reshape(self.bins).astype(FLOAT)
        self.Esw_norm       [:] = gist_data[:,12+offset].reshape(self.bins).astype(FLOAT)
        self.Eww_dens       [:] = gist_data[:,13+offset].reshape(self.bins).astype(FLOAT)
        self.Eww_norm_unref [:] = gist_data[:,14+offset].reshape(self.bins).astype(FLOAT)
        self.Dipole_x_dens  [:] = gist_data[:,15+offset].reshape(self.bins).astype(FLOAT)
        self.Dipole_y_dens  [:] = gist_data[:,16+offset].reshape(self.bins).astype(FLOAT)
        self.Dipole_z_dens  [:] = gist_data[:,17+offset].reshape(self.bins).astype(FLOAT)
        self.Dipole_dens    [:] = gist_data[:,18+offset].reshape(self.bins).astype(FLOAT)
        self.Neighbor_dens  [:] = gist_data[:,19+offset].reshape(self.bins).astype(FLOAT)
        self.Neighbor_norm  [:] = gist_data[:,20+offset].reshape(self.bins).astype(FLOAT)
        self.Order_norm     [:] = gist_data[:,21+offset].reshape(self.bins).astype(FLOAT)


class write_files(object):

    def __init__(self, Delta=None, Frac2Real=None, Bins=None, Origin=None, \
                 Value=None, XYZ=None, X=None, Y=None, Z=None, Format='PDB', \
                 Filename=None, Nan_fill=-1.0):

        """
        This class can write different file types.
        currently only dx and pdb are supported.
        """

        self._delta     = Delta
        self._frac2real = Frac2Real
        self._bins      = Bins
        self._origin    = Origin
        self._value     = Value
        self._x         = X
        self._y         = Y
        self._z         = Z
        self._format    = Format
        self._filename  = Filename
        self._xyz       = XYZ
        self._nan_fill  = Nan_fill

        if type(self._filename) != str:

            self._filename  = 'output.'
            self._filename  += self._format

        self._writers = {
                        'PDB'  : self._write_PDB,
                        'DX'   : self._write_DX,
                        }

        data = self._writers[self._format]()

        o = open(self._filename, "w")
        o.write(data)
        o.close()

    def _merge_x_y_z(self):

        return np.stack( ( self._x, self._y, self._z ), axis=1 )


    def _write_PDB(self):

        """
        Write a PDB file.
        This is intended for debugging. It writes all atoms
        as HETATM of element X with resname MAP.
        """

        if are_you_numpy(self._xyz):

            if self._xyz.shape[0] != 0:

                if self._xyz.shape[-1] != 3:

                    raise TypeError(
                        "XYZ array has wrong shape.")

        else:

            if not ( are_you_numpy(self._x) or are_you_numpy(self._y) or are_you_numpy(self._z) ):

                raise TypeError(
                    "If XYZ is not given, x,y and z coordinates\
                     must be given in separate arrays.")

            else:

                self._xyz = self._merge_x_y_z()

        if type(self._value) == type(None):

            self._value = np.zeros( len(self._xyz), dtype=float )

        data = 'REMARK File written by write_files class\n'

        for xyz_i, xyz in enumerate(self._xyz):

            #iterate over uppercase letters
            chain_id    = ascii_uppercase[int(len(str(xyz_i+1)) / 5 )]

            atom_counts = xyz_i - int(len(str(xyz_i+1)) / 6 ) * 100000
            resi_counts = xyz_i - int(len(str(xyz_i+1)) / 5 ) * 10000
            data += \
            '%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          \n' \
            %('HETATM',atom_counts+1,'X','', 'MAP', chain_id, resi_counts+1, '', xyz[0], xyz[1], xyz[2], 0.00, float( self._value[xyz_i] ) )

        data += 'END\n'

        return data


    def _write_DX(self):

        """
        Writes DX files according to openDX standard.
        """

        if not ( are_you_numpy(self._origin) or are_you_numpy(self._bins) ):

            raise TypeError(
            "Origin and bins must be given.")

        #This means not (a XOR b) or not (a or b)
        if are_you_numpy(self._delta) == are_you_numpy(self._frac2real) :

            raise TypeError(
            "Either delta or frac2real must be given.")

        if are_you_numpy(self._delta):

            self._frac2real = np.zeros((3,3), dtype=float)

            np.fill_diagonal(self._frac2real, self._delta)

        data = '''object 1 class gridpositions counts %d %d %d
origin %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
delta %8.4f %8.4f %8.4f
object 2 class gridconnections counts %d %d %d
object 3 class array type float rank 0 items %d data follows
''' %(self._bins[0], self._bins[1], self._bins[2],\
      self._origin[0], self._origin[1], self._origin[2],\
      self._frac2real[0][0], self._frac2real[0][1], self._frac2real[0][2],\
      self._frac2real[1][0], self._frac2real[1][1], self._frac2real[1][2],\
      self._frac2real[2][0], self._frac2real[2][1], self._frac2real[2][2],\
      self._bins[0],   self._bins[1],   self._bins[2],\
      self._bins[2] * self._bins[1] * self._bins[0])

        i = 0
        for x_i in range(0, self._bins[0]):
            for y_i in range(0, self._bins[1]):
                for z_i in range(0, self._bins[2]):
                    ### writing an integer instead of float
                    ### saves us some disk space
                    if np.isnan(self._value[x_i][y_i][z_i]):
                        data += str(self._nan_fill) + " "
                    else:
                        if self._value[x_i][y_i][z_i] == 0.0:
                            data += "0 " 
                        else:
                            data += str(self._value[x_i][y_i][z_i]) + ' '
                    i += 1
                    if i == 3:
                        data += '\n'
                        i     = 0
        return data


class PDB(object):

  """
  Class that reads a pdb file and provides pdb type data structure.
  """

  def __init__(self, Path):

    self.path = Path

    self.crd  = list()
    self.B    = list()

    with open(self.path, "r") as PDB_file:

      for i, line in enumerate(PDB_file):

        if not (line[0:6].rstrip() == 'ATOM' or line[0:6].rstrip() == 'HETATM'):

          continue

        if i <= 9999:

          #Coordinates
          self.crd.append(list())
          self.crd[-1].append(float(line.rstrip()[30:38]))
          self.crd[-1].append(float(line.rstrip()[38:46]))
          self.crd[-1].append(float(line.rstrip()[46:54]))

          #B-Factors
          self.B.append(line.rstrip()[54:59])

        if 9999 < i <= 99999:

          #Coordinates
          self.crd.append(list())
          self.crd[-1].append(float(line.rstrip()[31:39]))
          self.crd[-1].append(float(line.rstrip()[39:47]))
          self.crd[-1].append(float(line.rstrip()[47:55]))

          #B-Factors
          self.B.append(line.rstrip()[55:60])

        if i > 99999:

          #Coordinates
          self.crd.append(list())
          self.crd[-1].append(float(line.rstrip()[33:41]))
          self.crd[-1].append(float(line.rstrip()[41:49]))
          self.crd[-1].append(float(line.rstrip()[49:57]))

          #B-Factors
          self.B.append(line.rstrip()[57:62])

    self.crd  = np.array(self.crd)
    self.B    = np.array(self.B)