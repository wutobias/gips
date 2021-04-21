import socket
import datetime
import copy
from scipy import stats
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pygmo
from gips.utils.misc import mode_error

class print_fun(object):

    def __init__(self, fitter, mode=0, optimizer="basinhopping", 
                optparms=None, selection_A=None, selection_B=None, 
                prefix=None, verbose=False):

        self.fitter      = fitter
        self.mode        = mode
        self.optimizer   = optimizer
        self.optparms    = optparms
        self.selection_A = selection_A
        self.selection_B = selection_B
        self.prefix      = prefix
        self.verbose     = verbose

        self.len_A       = len(self.selection_A)
        self.len_B       = len(self.selection_B)

        self.pairs       = fitter.pairs
        self.parms       = fitter.parms

        if self.pairs:
            self.N_len = self.fitter.N_pairs
        else:
            self.N_len = self.fitter.N_case

        if self.mode==0:
            self.parmidx = list(range(self.parms))
        elif self.mode==1:
            self.parmidx = list(range(self.parms+1))
        elif self.mode==3:
            self.parmidx = list(range(self.parms))
        elif self.mode==4:
            self.parmidx = list(range(self.parms+1))
        elif self.mode==5:
            if self.pairs:
                self.parmidx = list(range(self.parms+1))
            else:
                self.parmidx = list(range(self.parms+2))
        elif self.mode==6:
            if self.pairs:
                self.parmidx = list(range(self.parms+2))
            else:
                self.parmidx = list(range(self.parms+3))
        elif self.mode==7:
            if self.pairs:
                self.parmidx = list(range(self.parms+4))
            else:
                self.parmidx = list(range(self.parms+7))
        else:
            mode_error(self.mode)

        if prefix == None:
            prefix=""
        elif type(prefix)!=str:
            raise TypeError("prefix must be of type str, but is of type %s" %type(prefix))

        self.exA = self.fitter._exp_data[self.selection_A]
        self.exB = self.fitter._exp_data[self.selection_B]

        self.x      = list()
        self.f_A    = list()
        self.f_B    = list()
        self.rmsd_A = list()
        self.rmsd_B = list()
        self.step   = list()
        self.R2_A   = list()
        self.R2_B   = list()

        self.__counter = 0

        self.init_output()


    def init_output(self):

        ### Open files for writing
        self.modelparms      = open("%sparms.out" %self.prefix, "w")
        self.modelprediction = open("%sprediction.out" %self.prefix, "w")
        self.modeldiff       = open("%sdiff.out" %self.prefix, "w")

        now = datetime.datetime.now()

        ### Start to write the headers
        ### The modelparms file
        self.modelparms.write("### File containing parameters of the functionals.\n")
        self.modelparms.write("### Written by gips.\n")
        self.modelparms.write("### Time: %s\n" %now.strftime("%Y-%m-%d %H:%M"))
        self.modelparms.write("### Hostname: %s\n" %socket.gethostname())
        self.modelparms.write("### The entropy contribution to free energy is calculated at 300 K\n")
        if isinstance(self.optparms, dict):
            for key, value in list(self.optparms.items()):
                self.modelparms.write("### %s: %s\n" %(key, value))
        self.modelparms.write("###\n")
        self.modelparms.write("### Step ")
        if self.parms==6:
            self.modelparms.write("E_aff[kcal/mol] ")
        elif self.parms==5:
            self.modelparms.write("Aff[kcal/mol] ")
        if self.mode==7:
            if not self.pairs:
                self.modelparms.write("e_co(Rec)[kcal/mol] ")
            self.modelparms.write("e_co(Cplx)[kcal/mol] ")
            self.modelparms.write("e_co(Lig)[kcal/mol] ")
        else:
            self.modelparms.write("e_co[kcal/mol] ")
        if self.parms==6:
            self.modelparms.write("S_aff[kcal/mol] ")
        if self.mode==7:
            if not self.pairs:
                self.modelparms.write("s_co(Rec)[kcal/mol] ")
            self.modelparms.write("s_co(Cplx)[kcal/mol] ")
            self.modelparms.write("s_co(Lig)[kcal/mol] ")
        else:
            self.modelparms.write("s_co[kcal/mol] ")
        if self.mode in [5,6,7]:
            if not self.pairs:
                self.modelparms.write("g_co(Rec)[1/Ang^3] ")
            self.modelparms.write("g_co(Cplx)[1/Ang^3] ")
            self.modelparms.write("g_co(Lig)[1/Ang^3] ")
        else:
            self.modelparms.write("g_co[1/Ang^3] ")
        if self.mode in [0,3,5]:
            self.modelparms.write("C[kcal/mol] ")
        elif self.mode in [1,4,6,7]:
            self.modelparms.write("C_E[kcal/mol] ")
            self.modelparms.write("C_S[kcal/mol] ")
        else:
            mode_error(self.mode)
        self.modelparms.write("SSE[kcal^2/mol^2](A) ")
        self.modelparms.write("SSE[kcal^2/mol^2](B) ")
        self.modelparms.write("R2(A) ")
        self.modelparms.write("R2(B) ")
        self.modelparms.write("rmsd[kcal/mol](A) ")
        self.modelparms.write("rmsd[kcal/mol](B) ")
        self.modelparms.write("\n")

        ### The model prediction file
        self.modelprediction.write("### File containing predicted values of the model.\n")
        self.modelprediction.write("### Written by gips.\n")
        self.modelprediction.write("### Time: %s\n" %now.strftime("%Y-%m-%d %H:%M"))
        self.modelprediction.write("### Hostname: %s\n" %socket.gethostname())
        self.modelprediction.write("### All units in [kcal/mol]\n")
        if isinstance(self.optparms, dict):
            for key, value in list(self.optparms.items()):
                self.modelprediction.write("### %s: %s\n" %(key, value))
        self.modelprediction.write("###\n")
        self.modelprediction.write("### Model values.\n")
        self.modelprediction.write("### All values in [kcal/mol].\n")
        if self.mode in [1,4,6,7]:
            if self.fitter.decomp_E:
                self.modelprediction.write("### First row for each step is free energy, second row is energy.\n")
            elif self.fitter.decomp_S:
                self.modelprediction.write("### First row for each step is free energy, second row is entropy.\n")
            else:
                self.modelprediction.write("### First row for each step is energy, second row is entropy.\n")
        self.modelprediction.write("#Title ")
        for name in self.fitter.name:
            self.modelprediction.write("%s "%name)
        self.modelprediction.write("\n")
        self.modelprediction.write("#Selection ")
        for i, name in enumerate(self.fitter.name):
            if i in self.selection_A:
                self.modelprediction.write("A ")
            elif i in self.selection_B:
                self.modelprediction.write("B ")
            else:
                raise KeyError("name=%s not found in selection A or selection B." %name)
        self.modelprediction.write("\n")
        if self.mode in [0,3,5]:
            self.modelprediction.write("-1 ")
            for value in self.fitter._exp_data:
                self.modelprediction.write("%6.3f "%value)
        elif self.mode in [1,4,6,7]:
            self.modelprediction.write("-1 ")
            for value in self.fitter._exp_data[:,0]:
                self.modelprediction.write("%6.3f "%value)
            self.modelprediction.write("\n")
            self.modelprediction.write("-1 ")
            for value in self.fitter._exp_data[:,1]:
                self.modelprediction.write("%6.3f "%value)
        else:
            mode_error(self.mode)
        self.modelprediction.write("\n")

        ### The model-experiment difference file
        self.modeldiff.write("### File containing differecnes between predicted and experimental values.\n")
        self.modeldiff.write("### Written by gips.\n")
        self.modeldiff.write("### Time: %s\n" %now.strftime("%Y-%m-%d %H:%M"))
        self.modeldiff.write("### Hostname: %s\n" %socket.gethostname())
        self.modeldiff.write("### All units in [kcal/mol]\n")
        if isinstance(self.optparms, dict):
            for key, value in list(self.optparms.items()):
                self.modeldiff.write("### %s: %s\n" %(key, value))
        self.modeldiff.write("###\n")
        self.modeldiff.write("### Model-Exp difference.\n")
        self.modeldiff.write("### All values in [kcal/mol].\n")
        if self.mode in [1,4,6,7]:
            if self.fitter.decomp_E:
                self.modelprediction.write("### First row for each step is free energy energy, second row is energy.\n")
            elif self.fitter.decomp_S:
                self.modelprediction.write("### First row for each step is free energy energy, second row is entropy.\n")
            else:
                self.modelprediction.write("### First row for each step is energy, second row is entropy.\n")
        self.modeldiff.write("Title ")
        
        for name in self.fitter.name:
            self.modeldiff.write("%s "%name)
        self.modeldiff.write("\n")
        self.modeldiff.write("# Selection ")
        for i, name in enumerate(self.fitter.name):
            if i in self.selection_A:
                self.modeldiff.write("A ")
            elif i in self.selection_B:
                self.modeldiff.write("B ")
            else:
                raise KeyError("name=%s not found in selection A or selection B." %name)
        self.modeldiff.write("\n")

        ### Flush file contents to disk
        self.modelparms.flush()
        self.modelprediction.flush()
        self.modeldiff.flush()


    def flush(self):
    
        self.modelparms.write("%d " %self.step[-1])
        self.modelprediction.write("%d " %self.step[-1])
        self.modeldiff.write("%d " %self.step[-1])

        if self.mode in [0,3,5]:
            for i in self.parmidx:
                self.modelparms.write("%6.3f " %self.x[-1][i])
            self.modelparms.write("%6.3f "     %self.f_A[-1][0])
            self.modelparms.write("%6.3f "     %self.f_B[-1][0])
            self.modelparms.write("%6.3f "     %self.R2_A[-1])
            self.modelparms.write("%6.3f "     %self.R2_B[-1])
            self.modelparms.write("%6.3f "     %self.rmsd_A[-1])
            self.modelparms.write("%6.3f "     %self.rmsd_B[-1])

        elif self.mode in [1,4,6,7]:
            ### Energy Output
            for i in self.parmidx:
                self.modelparms.write("%6.3f " %self.x[-1][i])
            self.modelparms.write("%6.3f "     %self.f_A[-1][0])
            self.modelparms.write("%6.3f "     %self.f_B[-1][0])
            self.modelparms.write("%6.3f "     %self.R2_A[-1][0])
            self.modelparms.write("%6.3f "     %self.R2_B[-1][0])
            self.modelparms.write("%6.3f "     %self.rmsd_A[-1][0])
            self.modelparms.write("%6.3f "     %self.rmsd_B[-1][0])
            self.modelparms.write("\n")

            ### Entropy Output
            self.modelparms.write("%d " %self.step[-1])
            for i in self.parmidx:
                self.modelparms.write("%6.3f " %self.x[-1][i])
            self.modelparms.write("%6.3f "     %self.f_A[-1][1])
            self.modelparms.write("%6.3f "     %self.f_B[-1][1])
            ### Note: This line is different then the one in energy output
            self.modelparms.write("%6.3f "     %self.R2_A[-1][1])
            self.modelparms.write("%6.3f "     %self.R2_B[-1][1])
            self.modelparms.write("%6.3f "     %self.rmsd_A[-1][1])
            self.modelparms.write("%6.3f "     %self.rmsd_B[-1][1])
        else:
            mode_error(self.mode)

        if self.mode in [0,3,5]:
            for i in range(self.N_len):
                self.modelprediction.write("%6.3f " %self.fitter._f[i])
                diff = self.fitter._exp_data[i] - self.fitter._f[i]
                self.modeldiff.write("%6.3f " %diff)
        elif self.mode in [1,4,6,7]:
            for i in range(self.N_len):
                self.modelprediction.write("%6.3f " %self.fitter._f[i,0])
                diff = self.fitter._exp_data[i,0] - self.fitter._f[i,0]
                self.modeldiff.write("%6.3f " %diff)
            self.modelprediction.write("\n")
            self.modelprediction.write("%d " %self.step[-1])
            self.modeldiff.write("\n")
            self.modeldiff.write("%d " %self.step[-1])
            for i in range(self.N_len):
                self.modelprediction.write("%6.3f " %self.fitter._f[i,1])
                diff = self.fitter._exp_data[i,1] - self.fitter._f[i,1]
                self.modeldiff.write("%6.3f " %diff)
        else:
            mode_error(self.mode)
            
        self.modelparms.write("\n")
        self.modelprediction.write("\n")
        self.modeldiff.write("\n")

        self.modelparms.flush()
        self.modelprediction.flush()
        self.modeldiff.flush()


    def finish(self):

        if self.optimizer=="brute":
            self.modelparms.write("\n")
            self.modelprediction.write("\n")
            self.modeldiff.write("\n")

            self.modelparms.close()
            self.modelprediction.close()
            self.modeldiff.close()

            return 1

        self.x      = np.array(self.x)
        self.f_A    = np.array(self.f_A)
        self.rmsd_A = np.array(self.rmsd_A)
        self.f_B    = np.array(self.f_B)
        self.rmsd_B = np.array(self.rmsd_B)

        self.R2_A   = np.array(self.R2_A)
        self.R2_B   = np.array(self.R2_B)

        ### Write out best result for selection A
        ### -------------------------------------

        if self.fitter.decomp:
            ### ndf (list of 1D NumPy int array): the non-dominated fronts
            ### dl  (list of 1D NumPy int array): the domination list
            ### dc  (1D NumPy int array)        : the domination count
            ### ndr (1D NumPy int array)        : the non-domination ranks
            ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(self.f_A)
            ax_A             = pygmo.plot_non_dominated_fronts(self.f_A)
            ax_A.figure.savefig("%spareto.selectionA.png" %self.prefix, dpi=1000)
            ax_A.figure.clear("all")

            ordered_ndf = list()
            for front in ndf:
                ordered_ndf.append(pygmo.sort_population_mo(self.f_A[front]))
        else:
            ordered_ndf = np.argsort(self.f_A, axis=0)

        self.modelparms.write("### Best result (A)\n")
        self.modelprediction.write("### Best result (A)\n")
        self.modeldiff.write("### Best result (A)\n")

        for front_count, front in enumerate(ordered_ndf):
            for solution_i in front:

                step   = self.step[solution_i]
                x      = self.x[solution_i]
                f_A    = self.f_A[solution_i]
                f_B    = self.f_B[solution_i]
                rmsd_A = self.rmsd_A[solution_i]
                rmsd_B = self.rmsd_B[solution_i]
                R2_A   = self.R2_A[solution_i]
                R2_B   = self.R2_B[solution_i]

                self.modelparms.write("%d/%d "      %(step, front_count))
                self.modelprediction.write("%d/%d " %(step, front_count))
                self.modeldiff.write("%d/%d "       %(step, front_count))

                self.fitter.gist_functional(x)
                self.fitter._f_process(x)

                if self.mode in [0,3,5]:
                    for i in self.parmidx:
                        self.modelparms.write("%6.3f " %x[i])
                    self.modelparms.write("%6.3f " %f_A[0])
                    self.modelparms.write("%6.3f " %f_B[0])
                    self.modelparms.write("%6.3f " %R2_A)
                    self.modelparms.write("%6.3f " %R2_B)
                    self.modelparms.write("%6.3f " %rmsd_A)
                    self.modelparms.write("%6.3f " %rmsd_B)

                elif self.mode in [1,4,6,7]:
                    ### Energy Output
                    for i in self.parmidx:
                        self.modelparms.write("%6.3f " %x[i])
                    self.modelparms.write("%6.3f " %f_A[0])
                    self.modelparms.write("%6.3f " %f_B[0])
                    self.modelparms.write("%6.3f " %R2_A[0])
                    self.modelparms.write("%6.3f " %R2_B[0])
                    self.modelparms.write("%6.3f " %rmsd_A[0])
                    self.modelparms.write("%6.3f " %rmsd_B[0])
                    self.modelparms.write("\n")

                    ### Entropy Output
                    self.modelparms.write("%d/%d " %(step, front_count))
                    for i in self.parmidx:
                        self.modelparms.write("%6.3f " %x[i])
                    self.modelparms.write("%6.3f " %f_A[1])
                    self.modelparms.write("%6.3f " %f_B[1])
                    self.modelparms.write("%6.3f " %R2_A[1])
                    self.modelparms.write("%6.3f " %R2_B[1])
                    self.modelparms.write("%6.3f " %rmsd_A[1])
                    self.modelparms.write("%6.3f " %rmsd_B[1])

                else:
                    mode_error(self.mode)

                if self.mode in [0,3,5]:
                    for i in range(self.N_len):
                        self.modelprediction.write("%6.3f " %self.fitter._f[i])
                        diff = self.fitter._exp_data[i] - self.fitter._f[i]
                        self.modeldiff.write("%6.3f " %diff)
                elif self.mode in [1,4,6,7]:
                    for i in range(self.N_len):
                        self.modelprediction.write("%6.3f " %self.fitter._f[i,0])
                        diff = self.fitter._exp_data[i,0] - self.fitter._f[i,0]
                        self.modeldiff.write("%6.3f " %diff)
                    self.modelprediction.write("\n")
                    self.modelprediction.write("%d/%d " %(step, front_count))
                    self.modeldiff.write("\n")
                    self.modeldiff.write("%d/%d "       %(step, front_count))
                    for i in range(self.N_len):
                        self.modelprediction.write("%6.3f " %self.fitter._f[i,1])
                        diff = self.fitter._exp_data[i,1] - self.fitter._f[i,1]
                        self.modeldiff.write("%6.3f " %diff)
                else:
                    mode_error(self.mode)

                self.modelparms.write("\n")
                self.modelprediction.write("\n")
                self.modeldiff.write("\n")

        self.modelparms.write("\n")
        self.modelprediction.write("\n")
        self.modeldiff.write("\n")

        self.modelparms.close()
        self.modelprediction.close()
        self.modeldiff.close()


    def get_stats(self):

        if self.mode in [0,3,5]:

            slope_A, intercept_A, r_value_A, p_value_A, std_err_A = \
             stats.linregress(self.fitter._f[self.selection_A], self.exA)
            slope_B, intercept_B, r_value_B, p_value_B, std_err_B = \
             stats.linregress(self.fitter._f[self.selection_B], self.exB)

            R2_A = r_value_A**2
            R2_B = r_value_B**2

            return R2_A, R2_B

        elif self.mode in [1,4,6,7]:

            ### Selection A Energy
            slope_A_E, intercept_A_E, r_value_A_E, p_value_A_E, std_err_A_E = \
             stats.linregress(self.fitter._f[self.selection_A,0], self.exA[:,0])
            ### Selection A Entropy
            slope_A_S, intercept_A_S, r_value_A_S, p_value_A_S, std_err_A_S = \
             stats.linregress(self.fitter._f[self.selection_A,1], self.exA[:,1])

            ### Selection B Energy
            slope_B_E, intercept_B_E, r_value_B_E, p_value_B_E, std_err_B_E = \
             stats.linregress(self.fitter._f[self.selection_B,0], self.exB[:,0])
            ### Selection B Entropy
            slope_B_S, intercept_B_S, r_value_B_S, p_value_B_S, std_err_B_S = \
             stats.linregress(self.fitter._f[self.selection_B,1], self.exB[:,1])

            R2_A = [r_value_A_E**2, r_value_A_S**2]
            R2_B = [r_value_B_E**2, r_value_B_S**2]

            return R2_A, R2_B

        else:
            mode_error(self.mode)


    def __call__(self, x):

        self.x.append(copy.copy(x))

        self.fitter.gist_functional(x)
        self.fitter._f_process(x)

        self.f_A.append(copy.copy(self.fitter._f_error_select(x, self.selection_A)))
        self.rmsd_A.append(copy.copy(self.fitter.rmsd_select(self.selection_A)))
        self.f_B.append(copy.copy(self.fitter._f_error_select(x, self.selection_B)))
        self.rmsd_B.append(copy.copy(self.fitter.rmsd_select(self.selection_B)))

        R2_A, R2_B = self.get_stats()
        self.R2_A.append(copy.copy(R2_A))
        self.R2_B.append(copy.copy(R2_B))

        self.step.append(copy.copy(self.__counter))
        self.__counter += 1