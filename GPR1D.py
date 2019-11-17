"""
Classes for Gaussian Process Regression fitting of 1D data with errorbars. Built in Python 3.x, adapted to be Python 2.x compatible.

These classes were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET,
in
IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis (Boston, MA, 2017),
`<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

"""
#    Kernel theory: "Gaussian Process for Machine Learning", C.E. Rasmussen, C.K.I. Williams (2006)
#    Gaussian process theory: "Gaussian Processes for Machine Learning", C.E. Rasmussen and C.K.I. Williams (2006)

# Required imports
import warnings
import re
import copy
import numpy as np
import scipy.special as spsp
import scipy.linalg as spla
import scipy.stats as spst
from operator import itemgetter

np_itypes = (np.int8,np.int16,np.int32,np.int64)
np_utypes = (np.uint8,np.uint16,np.uint32,np.uint64)
np_ftypes = (np.float16,np.float32,np.float64)
array = np.ndarray

__all__ = ['Sum_Kernel', 'Product_Kernel', 'Symmetric_Kernel',  # Kernel operator classes
           'Constant_Kernel', 'Noise_Kernel', 'Linear_Kernel', 'Poly_Order_Kernel', 'SE_Kernel', 'RQ_Kernel',
           'Matern_HI_Kernel', 'NN_Kernel', 'Gibbs_Kernel',  # Kernel classes
           'Constant_WarpingFunction', 'IG_WarpingFunction',  # Warping function classes for Gibbs Kernel
           'KernelConstructor', 'KernelReconstructor',  # Kernel construction functions
           'GaussianProcessRegression1D']  # Main interpolation class


class _Kernel(object):
    """
    Base class to be inherited by **ALL** kernel implementations in order for type checks to succeed.
    Type checking done with :code:`isinstance(<object>,<this_module>._Kernel)`.

    Ideology:

    - :code:`self._fname` is a string, designed to provide an easy way to check the kernel instance type.
    - :code:`self._function` contains the covariance function, k, along with **at least** dk/dx1, dk/dx2, and d^2k/dx1dx2.
    - :code:`self._hyperparameters` contains free variables that are designed to vary in logarithmic-space.
    - :code:`self._constants` contains free variables that should not be changed during parameter searches, or true constants.
    - :code:`self._bounds` contains the bounds of the free variables to be used in randomized kernel restart algorithms.

    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly **NOT** recommended unless you are familiar with how these structures work and their interdependencies.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    :kwarg name: str. Codename of :code:`_Kernel` class implementation.

    :kwarg func: callable. Covariance function of :code:`_Kernel` class implementation.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg hyps: array. Hyperparameters to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

    :kwarg csts: array. Constants to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

    :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_Kernel` class implementation. (optional)

    :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_Kernel` class implementation. (optional)
    """

    def __init__(self,name=None,func=None,hderf=False,hyps=None,csts=None,htags=None,ctags=None):
        """
        Initializes the :code:`_Kernel` instance.

        .. note::

            Nothing is done with the :code:`htags` and :code:`ctags` arguments currently. (v >= 1.0.1)

        :kwarg name: str. Codename of :code:`_Kernel` class implementation.

        :kwarg func: callable. Covariance function of :code:`_Kernel` class implementation.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

        :kwarg csts: array. Constants to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

        :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_Kernel` class implementation. (optional)

        :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_Kernel` class implementation. (optional)

        :returns: none.
        """

        self._fname = name
        self._function = func if callable(func) else None
        self._hyperparameters = np.array(hyps,dtype=np.float64).flatten() if isinstance(hyps,(list,tuple,array)) else None
        self._constants = np.array(csts,dtype=np.float64).flatten() if isinstance(csts,(list,tuple,array)) else None
        self._hyp_lbounds = None
        self._hyp_ubounds = None
        self._hderflag = hderf
        self._force_bounds = False


    def __call__(self,x1,x2,der=0,hder=None):
        """
        Default class call function, evaluates the stored covariance function at the input values.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        k_out = None
        if callable(self._function):
            xt1 = None
            xt2 = None
            dert = 0
            hdert = None
            if isinstance(x1,(float,int,np_itypes,np_utypes,np_ftypes)):
                xt1 = np.array(np.atleast_2d(x1),dtype=np.float64)
            elif isinstance(x1,(list,tuple,array)):
                xt1 = np.array(np.atleast_2d(x1),dtype=np.float64)
            if isinstance(x2,(float,int,np_itypes,np_utypes,np_ftypes)):
                xt2 = np.array(np.atleast_2d(x2),dtype=np.float64)
            elif isinstance(x2,(list,tuple,array)):
                xt2 = np.array(np.atleast_2d(x2),dtype=np.float64)
            if isinstance(der,(float,int,np_itypes,np_utypes,np_ftypes)):
                dert = int(der)
            if isinstance(hder,(float,int,np_itypes,np_utypes,np_ftypes)):
                hdert = int(hder)
            if isinstance(xt1,array) and isinstance(xt2,array):
                k_out = self._function(xt1,xt2,dert,hdert)
            else:
                raise TypeError('Arguments x1 and x2 must be a 2D-array-like object.')
        else:
            raise NotImplementedError('Covariance function of %s Kernel object not yet defined.' % (self.name))
        return k_out


    def __eq__(self,other):
        """
	Custom equality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_Kernel` class instance.

        :return: bool. Indicates whether the two objects are equal to each other.
        """
        status = False
        if isinstance(other,_Kernel):
            if self.name == other.name:
                shyp = np.all(np.isclose(self.hyperparameters,other.hyperparameters))
                scst = np.all(np.isclose(self.constants,other.constants))
                status = self.hyperparameters.size == other.hyperparameters.size and self.constants.size == other.constants.size and shyp and scst
        return status


    def __ne__(self,other):
        """
	Custom inequality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_Kernel` class instance.

        :return: bool. Indicates whether the two objects are not equal to each other.
        """
        return not self.__eq__(other)


    def enforce_bounds(self,value=True):
        """
        Sets a flag to enforce the given hyperparameter bounds.

        :kwarg value: bool. Boolean value to set the flag.

        :reutnrs: none.
        """
        self._force_bounds = True if value else False


    @property
    def name(self):
        """
        Returns the codename of the :code:`_Kernel` instance.

        :returns: str. Codename of the :code:`_Kernel` instance.
        """

        val = self._fname if isinstance(self._fname,str) else "None"
        return val


    @property
    def hyperparameters(self):
        """
        Return the hyperparameters stored in the :code:`_Kernel` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_Kernel` class implementation.
        """

        val = np.array([])
        if isinstance(self._hyperparameters,array):
            val = self._hyperparameters.copy()
        return val


    @property
    def constants(self):
        """
        Return the constants stored in the :code:`_Kernel` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_Kernel` class implementation.
        """

        val = np.array([])
        if isinstance(self._constants,array):
            val = self._constants.copy()
        return val


    @property
    def bounds(self):
        """
        Return the hyperparameter search bounds stored in the :code:`_Kernel` instance.

        :returns: array. Hyperparameter lower/upper bounds list, ordered according to the specific :code:`_Kernel` class implementation.
        """

        val = None
        if isinstance(self._hyp_lbounds,array) and isinstance(self._hyp_ubounds,array) and self._hyp_lbounds.shape == self._hyp_ubounds.shape:
            val = np.vstack((self._hyp_lbounds.flatten(),self._hyp_ubounds.flatten()))
        return val


    def is_hderiv_implemented(self):
        """
        Checks if the explicit hyperparameter derivative is implemented in the :code:`_Kernel` class implementation.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        """

        return self._hderflag


    @hyperparameters.setter
    def hyperparameters(self,theta):
        """
        Set the hyperparameters stored in the :code:`_Kernel` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_Kernel` implementation.

        :returns: none.
        """

        userhyps = None
        if isinstance(theta,(list,tuple,array)):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError('%s Kernel hyperparameters must be given as an array-like object.' % (self._fname))
        if isinstance(self._hyperparameters,array):
            if userhyps.size >= self._hyperparameters.size:
                if self._force_bounds and isinstance(self._hyp_lbounds,array) and self._hyp_lbounds.size == self._hyperparameters.size:
                    htemp = userhyps[:self._hyperparameters.size]
                    lcheck = (htemp < self._hyp_lbounds)
                    htemp[lcheck] = self._hyp_lbounds[lcheck]
                    userhyps[:self._hyperparameters.size] = htemp
                if self._force_bounds and isinstance(self._hyp_ubounds,array) and self._hyp_ubounds.size == self._hyperparameters.size:
                    htemp = userhyps[:self._hyperparameters.size]
                    ucheck = (htemp > self._hyp_ubounds)
                    htemp[ucheck] = self._hyp_ubounds[ucheck]
                    userhyps[:self._hyperparameters.size] = htemp
                self._hyperparameters = np.array(userhyps[:self._hyperparameters.size],dtype=np.float64)
            else:
                raise ValueError('%s Kernel hyperparameters must contain at least %d elements.' % (self.name,self._hyperparameters.size))
        else:
            warnings.warn('%s Kernel instance has no hyperparameters.' % (self.name),stacklevel=2)


    @constants.setter
    def constants(self,consts):
        """
        Set the constants stored in the :code:`_Kernel` instance.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_Kernel` class implementation.

        :returns: none.
        """

        usercsts = None
        if isinstance(consts,(list,tuple,array)):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError('%s Kernel constants must be given as an array-like object.' % (self._fname))
        if isinstance(self._constants,array):
            if usercsts.size >= self._constants.size:
                self._constants = np.array(usercsts[:self._constants.size],dtype=np.float64)
            else:
                raise ValueError('%s Kernel constants must contain at least %d elements.' % (self.name,self._constants.size))
        else:
            warnings.warn('%s Kernel instance has no constants.' % (self.name),stacklevel=2)


    @bounds.setter
    def bounds(self,bounds):
        """
        Set the hyperparameter bounds stored in the :code:`_Kernel` instance.

        :arg bounds: 2D array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_Kernel` class implementation.

        :returns: none.
        """

        userbnds = None
        if isinstance(bounds,(list,tuple,array)):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError('%s Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        if userbnds.shape[0] != 2:
            raise TypeError('%s Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        if isinstance(self._hyperparameters,array):
            if userbnds.shape[1] >= self._hyperparameters.size:
                self._hyp_lbounds = np.array(userbnds[0,:self._hyperparameters.size],dtype=np.float64)
                self._hyp_ubounds = np.array(userbnds[1,:self._hyperparameters.size],dtype=np.float64)
                if self._force_bounds:
                    self.hyperparameters = self._hyperparameters.copy()
            else:
                raise ValueError('%s Kernel bounds must be a 2D-array-like object with exactly 2 rows and at least %d elements per row.' % (self.name,self._hyperparameters.size))
        else:
            warnings.warn('%s Kernel instance has no hyperparameters to set bounds for.' % (self.name),stacklevel=2)



class _OperatorKernel(_Kernel):
    """
    Base operator class to be inherited by **ALL** operator kernel implementations for custom get/set functions.
    Type checking done with :code:`isinstance(<object>,<this_module>._OperatorKernel)` if needed.

    Ideology:

    - :code:`self._kernel_list` is a Python list of :code:`_Kernel` instances on which the specified operation will be performed

    Get/set functions adjusted to call get/set functions of each constituent kernel instead of using its own
    attributes, which are mostly left as :code:`None`.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    :kwarg name: str. Codename of :code:`_OperatorKernel` class implementation.

    :kwarg func: callable. Covariance function of :code:`_OperatorKernel` class implementation, ideally an operation on provided :code:`_Kernel` instances.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg klist: array. List of :code:`_Kernel` instances to be operated on by the :code:`_OperatorKernel` instance, input order determines the order of parameter lists.
    """

    def __init__(self,name="None",func=None,hderf=False,klist=None):
        """
        Initializes the :code:`_OperatorKernel` instance.

        :kwarg name: str. Codename of :code:`_OperatorKernel` class implementation.

        :kwarg func: callable. Covariance function of :code:`_OperatorKernel` class implementation, ideally an operation on provided :code:`_Kernel` instances.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg klist: array. List of :code:`_Kernel` instances to be operated on by the :code:`_OperatorKernel` instance, input order determines the order of parameter lists.

        :returns: none.
        """
        super(_OperatorKernel,self).__init__(name,func,hderf)
        self._kernel_list = klist if klist is not None else []


    @property
    def name(self):
        """
        Returns the codename of the :code:`_OperatorKernel` instance.

        :returns: str. Codename of the :code:`_OperatorKernel` instance.
        """

        val = self._fname if isinstance(self._fname,str) else None
        temp = ""
        for kk in self._kernel_list:
            temp = temp + "-" + kk.name if temp else kk.name
        temp = "(" + temp + ")"
        val = val + temp if val is not None else "None"
        return val


    @property
    def basename(self):
        """
        Returns the base codename of the :code:`_OperatorKernel` instance.

        :returns: str. Base codename of the :code:`_OperatorKernel` instance.
        """

        val = self._fname if isinstance(self._fname,str) else "None"
        return val


    @property
    def hyperparameters(self):
        """
        Return the hyperparameters of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        """

        val = np.array([])
        for kk in self._kernel_list:
            val = np.hstack((val,kk.hyperparameters))
        return val


    @property
    def constants(self):
        """
        Return the constants of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        """

        val = np.array([])
        for kk in self._kernel_list:
            val = np.hstack((val,kk.constants))
        return val


    @property
    def bounds(self):
        """
        Return the hyperparameter bounds of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Hyperparameter lower/upper bounds list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        """

        val = None
        for kk in self._kernel_list:
            kval = kk.bounds
            if kval is not None:
                val = np.hstack((val,kval)) if val is not None else kval.copy()
        return val


    @hyperparameters.setter
    def hyperparameters(self,theta):
        """
        Set the hyperparameters stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        """

        userhyps = None
        if isinstance(theta,(list,tuple,array)):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError('%s OperatorKernel hyperparameters must be given as an array-like object.' % (self._fname))
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userhyps.size >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.hyperparameters.size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.hyperparameters = theta[ndone:]
                        else:
                            kk.hyperparameters = theta[ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError('%s OperatorKernal hyperparameters must contain at least %d elements.' % (self.name,nhyps))
        else:
            warnings.warn('%s OperatorKernel instance has no hyperparameters.' % (self.name),stacklevel=2)


    @constants.setter
    def constants(self,consts):
        """
        Set the constants stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        """

        usercsts = None
        if isinstance(consts,(list,tuple,array)):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError('%s OperatorKernel constants must be given as an array-like object.' % (self._fname))
        ncsts = self.constants.size
        if ncsts > 0:
            if usercsts.size >= ncsts:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.constants.size
                    if nhere != ndone:
                        if nhere == ncsts:
                            kk.constants = consts[ndone:]
                        else:
                            kk.constants = consts[ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError('%s OperatorKernel constants must contain at least %d elements.' % (self.name,ncsts))
        else:
            warnings.warn('%s OperatorKernel instance has no constants.' % (self.name),stacklevel=2)


    @bounds.setter
    def bounds(self,bounds):
        """
        Set the hyperparameter bounds stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        """

        userbnds = None
        if isinstance(bounds,(list,tuple,array)):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError('%s OperatorKernel bounds must be given as a 2d-array-like object with exactly 2 rows.' % (self._fname))
        if userbnds.shape[0] != 2:
            raise TypeError('%s OperatorKernel bounds must be given as a 2d-array-like object with exactly 2 rows.' % (self._fname))
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userbnds.shape[1] >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.hyperparameters.size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.bounds = userbnds[:,ndone:]
                        else:
                            kk.bounds = userbnds[:,ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError('%s OperatorKernel bounds must be a 2D-array-like object with exactly 2 rows and contain at least %d elements per row.' % (self.name,self._hyperparameters.size))
        else:
            warnings.warn('%s OperatorKernel instance has no hyperparameters to set bounds for.' % (self.name),stacklevel=2)



class _WarpingFunction(object):
    """
    Base class to be inherited by **ALL** warping function implementations in order for type checks to succeed.
    Type checking done with :code:`isinstance(<object>,<this_module>._WarpingFunction)`.

    Ideology:

    - :code:`self._fname` is a string, designed to provide an easy way to check the warping function instance type.
    - :code:`self._function` contains the warping function, l, along with *at least* dl/dz and d^2l/dz^2.
    - :code:`self._hyperparameters` contains free variables that are designed to vary in logarithmic-space.
    - :code:`self._constants` contains free variables that should not be changed during hyperparameter optimization, or true constants.
    - :code:`self._bounds` contains the bounds of the free variables to be used in randomized kernel restarts.

    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly **NOT** recommended unless you are familiar with how these structures work and their interdependencies.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    .. note::

        The usage of the variable z in the documentation is simply to emphasize the generality of the object. In actuality,
        it is the same as x within the :code:`_Kernel` base class.

    :kwarg name: str. Codename of :code:`_WarpingFunction` class implementation.

    :kwarg func: callable. Warping function of :code:`_WarpingFunction` class implementation.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg hyps: array. Hyperparameters to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

    :kwarg csts: array. Constants to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

    :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

    :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)
    """

    def __init__(self,name="None",func=None,hderf=False,hyps=None,csts=None,htags=None,ctags=None):
        """
        Initializes the :code:`_WarpingFunction` instance.

        .. note::

            Nothing is done with the :code:`htags` and :code:`ctags` arguments currently. (v >= 1.0.1)

        :kwarg name: str. Codename of :code:`_WarpingFunction` class implementation.

        :kwarg func: callable. Warping function of :code:`_WarpingFunction` class implementation.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :kwarg csts: array. Constants to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

        :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

        :returns: none.
        """

        self._fname = name
        self._function = func if func is not None else None
        self._hyperparameters = np.array(hyps,dtype=np.float64) if isinstance(hyps,(list,tuple,array)) else None
        self._constants = np.array(csts,dtype=np.float64) if isinstance(csts,(list,tuple,array)) else None
        self._hyp_lbounds = None
        self._hyp_ubounds = None
        self._hderflag = hderf
        self._force_bounds = False


    def __call__(self,zz,der=0,hder=None):
        """
        Default class call function, evaluates the stored warping function at the input values.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        """

        k_out = None
        if self._function is not None:
            k_out = self._function(zz,der,hder)
        else:
            raise NotImplementedError('Warping function not yet defined.')
        return k_out


    def __eq__(self,other):
        """
	Custom equality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_WarpingFunction` class instance.

        :return: bool. Indicates whether the two objects are equal to each other.
        """
        status = False
        if isinstance(other,_WarpingFunction):
            if self.name == other.name:
                shyp = np.all(np.isclose(self.hyperparameters,other.hyperparameters))
                scst = np.all(np.isclose(self.constants,other.constants))
                status = self.hyperparameters.size == other.hyperparameters.size and self.constants.size == other.constants.size and shyp and scst
        return status


    def __ne__(self,other):
        """
	Custom inequality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_WarpingFunction` class instance.

        :return: bool. Indicates whether the two objects are not equal to each other.
        """
        return not self.__eq__(other)


    def enforce_bounds(self,value=True):
        """
        Sets a flag to enforce the given hyperparameter bounds.

        :kwarg value: bool. Boolean value to set the flag.

        :returns: none.
        """
        self._force_bounds = True if value else False


    @property
    def name(self):
        """
        Returns the codename of the :code:`_WarpingFunction` instance.

        :returns: str. Codename of the :code:`_WarpingFunction` instance.
        """

        val = self._fname if isinstance(self._fname,str) else "None"
        return val


    @property
    def hyperparameters(self):
        """
        Return the hyperparameters stored in the :code:`_WarpingFunction` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        """

        val = np.array([])
        if self._hyperparameters is not None:
            val = self._hyperparameters.copy()
        return val


    @property
    def constants(self):
        """
        Return the constants stored in the :code:`_WarpingFunction` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        """

        val = np.array([])
        if self._constants is not None:
            val = self._constants.copy()
        return val


    @property
    def bounds(self):
        """
        Return the hyperparameter search bounds stored in the :code:`_WarpingFunction` instance.

        :returns: array. Hyperparameter bounds list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        """

        val = None
        if isinstance(self._hyp_lbounds,array) and isinstance(self._hyp_ubounds,array) and self._hyp_lbounds.shape == self._hyp_ubounds.shape:
            val = np.vstack((self._hyp_lbounds.flatten(),self._hyp_ubounds.flatten()))
        return val


    def is_hderiv_implemented(self):
        """
        Checks if the explicit hyperparameter derivative is implemented in this :code:`_WarpingFunction` class implementation.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        """

        return self._hderflag


    @hyperparameters.setter
    def hyperparameters(self,theta):
        """
        Set the hyperparameters stored in the :code:`_WarpingFunction` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        userhyps = None
        if isinstance(theta,(list,tuple,array)):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError('%s WarpingFunction hyperparameters must be given as an array-like object.' % (self._fname))
        if isinstance(self._hyperparameters,array):
            if userhyps.size >= self._hyperparameters.size:
                if self._force_bounds and isinstance(self._hyp_lbounds,array) and self._hyp_lbounds.size == self._hyperparameters.size:
                    htemp = userhyps[:self._hyperparameters.size]
                    lcheck = (htemp < self._hyp_lbounds)
                    htemp[lcheck] = self._hyp_lbounds[lcheck]
                    userhyps[:self._hyperparameters.size] = htemp
                if self._force_bounds and isinstance(self._hyp_ubounds,array) and self._hyp_ubounds.size == self._hyperparameters.size:
                    htemp = userhyps[:self._hyperparameters.size]
                    ucheck = (htemp > self._hyp_ubounds)
                    htemp[ucheck] = self._hyp_ubounds[ucheck]
                    userhyps[:self._hyperparameters.size] = htemp
                self._hyperparameters = np.array(userhyps[:self._hyperparameters.size],dtype=np.float64)
            else:
                raise ValueError('%s WarpingFunction hyperparameters must contain at least %d elements.' % (self.name,self._hyperparameters.size))
        else:
            warnings.warn('%s WarpingFunction instance has no hyperparameters.' % (self.name),stacklevel=2)


    @constants.setter
    def constants(self,consts):
        """
        Set the constants stored in the :code:`_WarpingFunction` object.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        usercsts = None
        if isinstance(consts,(list,tuple,array)):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError('%s WarpingFunction constants must be given as an array-like object.' % (self._fname))
        if isinstance(self._constants,array):
            if usercsts.size >= self._constants.size:
                self._constants = np.array(usercsts[:self._constants.size],dtype=np.float64)
            else:
                raise ValueError('%s WarpingFunction constants must contain at least %d elements.' % (self.name,self._constants.size))
        else:
            warnings.warn('%s WarpingFunction instance has no constants.' % (self.name),stacklevel=2)


    @bounds.setter
    def bounds(self,bounds):
        """
        Set the hyperparameter bounds stored in the :code:`_WarpingFunction` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        userbnds = None
        if isinstance(bounds,(list,tuple,array)):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError('%s WarpingFunction bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        if userbnds.shape[0] != 2:
            raise TypeError('%s WarpingFunction bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        if isinstance(self._hyperparameters,array):
            if userbnds.shape[1] >= self._hyperparameters.size:
                self._hyp_lbounds = np.array(userbnds[0,:self._hyperparameters.size],dtype=np.float64)
                self._hyp_ubounds = np.array(userbnds[1,:self._hyperparameters.size],dtype=np.float64)
                if self._force_bounds:
                    self.hyperparameters = self._hyperparameters.copy()
            else:
                raise ValueError('%s WarpingFunction bounds must be a 2D-array-like object with exactly 2 rows and contain at least %d elements per row.' % (self.name,self._hyperparameters.size))
        else:
            warnings.warn('%s WarpingFunction instance has no hyperparameters to set bounds for.' % (self.name),stacklevel=2)



# ****************************************************************************************************************************************
# ------- Place ALL custom kernel implementations BELOW ----------------------------------------------------------------------------------
# ****************************************************************************************************************************************

class Sum_Kernel(_OperatorKernel):
    """
    Sum Kernel: Implements the sum of two (or more) separate kernels.

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be added together. Must provide a minimum of 2.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be added together. Must contain a minimum of 2.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2 values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        covm = np.full(x1.shape,np.NaN) if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp)
            if ihyp is not None:
                nhyps = kk.hyperparameters.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the :code:`Sum_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be added together. Must provide a minimum of 2.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be added together. Must contain a minimum of 2.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        if len(args) >= 2 and isinstance(args[0],_Kernel) and isinstance(args[1],_Kernel):
            for kk in args:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
        elif isinstance(klist,list) and len(klist) >= 2 and isinstance(klist[0],_Kernel) and isinstance(klist[1],_Kernel):
            for kk in klist:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel instances.')
        super(Sum_Kernel,self).__init__("Sum",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Sum_Kernel(klist=kcopy_list)
        return kcopy



class Product_Kernel(_OperatorKernel):
    """
    Product Kernel: Implements the product of two (or more) separate kernels.

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be multiplied together. Must provide a minimum of 2.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be multiplied together. Must contain a minimum of 2.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        covm = np.full(x1.shape,np.NaN) if self._kernel_list is None else np.zeros(x1.shape)
        nks = len(self._kernel_list) if self._kernel_list is not None else 0
        dermat = np.atleast_2d([0] * nks)
        sd = int(np.sign(der))
        for ii in np.arange(0,int(sd * der)):
            for jj in np.arange(1,nks):
                deradd = dermat.copy()
                dermat = np.vstack((dermat,deradd))
            for row in np.arange(0,dermat.shape[0]):
                rem = row % nks
                fac = (row - rem) / (nks**int(sd * der))
                idx = int((rem + fac) % nks)
                dermat[row,idx] = dermat[row,idx] + 1
        oddfilt = (np.mod(dermat,2) != 0)
        dermat[oddfilt] = sd * dermat[oddfilt]
        for row in np.arange(0,dermat.shape[0]):
            ihyp = hder
            covterm = np.ones(x1.shape)
            for col in np.arange(0,dermat.shape[1]):
                kk = self._kernel_list[col]
                covterm = covterm * kk(x1,x2,dermat[row,col],ihyp)
                if ihyp is not None:
                    nhyps = kk.hyperparameters.size
                    ihyp = ihyp - nhyps
            covm = covm + covterm
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the :code:`Product_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be multiplied together. Must provide a minimum of 2.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be multiplied together. Must contain a minimum of 2.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        name = "None"
        if len(args) >= 2 and isinstance(args[0],_Kernel) and isinstance(args[1],_Kernel):
            name = ""
            for kk in args:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)

        elif isinstance(klist,list) and len(klist) >= 2 and isinstance(klist[0],_Kernel) and isinstance(klist[1],_Kernel):
            name = ""
            for kk in klist:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.name if name else kk.name
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel objects.')
        super(Product_Kernel,self).__init__("Prod("+name+")",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Product_Kernel(klist=kcopy_list)
        return kcopy



class Symmetric_Kernel(_OperatorKernel):
    """
    1D Symmetric Kernel: Enforces even symmetry about zero for any given kernel. Although
    this class accepts multiple arguments, it only uses first :code:`_Kernel` argument.

    This is really only useful if you wish to rigourously infer data on other side of axis
    of symmetry without assuming the data can just be flipped or if data exists on other
    side but a symmetric solution is desired. **This capability is NOT fully tested!**

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be given flip symmetry. Must provide a minimum of 1.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be given flip symmetry. Must contain a minimum of 1.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the covariance function at, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        covm = np.full(x1.shape,np.NaN) if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp) + kk(-x1,x2,der,ihyp)      # Not sure if division by 2 is necessary to conserve covm
            if ihyp is not None:
                nhyps = kk.hyperparameters.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the :code:`Symmetric_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be given flip symmetry. Must provide a minimum of 1.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be given flip symmetry. Must contain a minimum of 1.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        name = "None"
        if len(args) >= 1 and isinstance(args[0],_Kernel):
            name = ""
            if len(args) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = args[0]
            uklist.append(kk)
            name = name + kk.name
        elif isinstance(klist,list) and len(klist) >= 1 and isinstance(klist[0],_Kernel):
            name = ""
            if len(klist) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = klist[0]
            uklist.append(kk)
            name = name + kk.name
        else:
            raise TypeError('Arguments to Symmetric_Kernel must be Kernel objects.')
        super(Symmetric_Kernel,self).__init__("Sym("+name+")",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Symmetric_Kernel(klist=kcopy_list)
        return kcopy



class Constant_Kernel(_Kernel):
    """
    Constant Kernel: always evaluates to a constant value, regardless of input pair.

    .. warning::

        This is **NOT inherently a valid covariance function**, as it yields
        singular covariance matrices! However, it provides an alternate way
        to enforce or relax the fit smoothness. **This capability is NOT fully
        tested!**

    :kwarg cv: float. Constant value which kernel always evaluates to.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        .. warning::

            This is **not** inherently a valid covariance function, as it results
            in singular matrices!

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        c_hyp = csts[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = c_hyp * np.ones(rr.shape)
            elif hder == 0:
                covm = np.ones(rr.shape)
        return covm


    def __init__(self,cv=1.0):
        """
        Initializes the :code:`Constant_Kernel` instance.

        :kwarg cv: float. Constant value which kernel always evaluates to.

        :returns: none.
        """

        csts = np.zeros((1,))
        if isinstance(cv,(float,int,np_itypes,np_utypes,np_ftypes)):
            csts[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super(Constant_Kernel,self).__init__("C",self.__calc_covm,True,None,csts)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(csts[0])
        kcopy = Constant_Kernel(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Noise_Kernel(_Kernel):
    """
    Noise Kernel: adds a user-defined degree of expected noise in the GPR regression, emulates a
    constant assumed fit noise level.

    .. note::

        The noise implemented by this kernel is **conceptually not the same** as measurement error,
        which should be applied externally in GP regression implementation!!!

    :kwarg nv: float. Hyperparameter representing the noise level.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1 values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2 values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        n_hyp = hyps[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm[rr == 0.0] = n_hyp**2.0
            elif hder == 0:
                covm[rr == 0.0] = 2.0 * n_hyp
#       Applied second derivative of Kronecker delta, assuming it is actually a Gaussian centred on rr = 0 with small width, ss
#       Surprisingly provides good variance estimate but issues with enforcing derivative constraints (needs more work!)
#        Commented out for stability reasons.
#        elif der == 2 or der == -2:
#            drdx1 = np.sign(x1 - x2)
#            drdx1[drdx1==0] = 1.0
#            drdx2 = np.sign(x2 - x1)
#            drdx2[drdx2==0] = -1.0
#            trr = rr[rr > 0.0]
#            ss = 0.0 if trr.size == 0 else np.nanmin(trr)
#            if hder is None:
#                covm[rr == 0.0] = -drdx1[rr == 0.0] * drdx2[rr == 0.0] * 2.0 * n_hyp**2.0 / ss**2.0
#            elif hder == 0:
#                covm[rr == 0.0] = -drdx1[rr == 0.0] * drdx2[rr == 0.0] * 4.0 * n_hyp / ss**2.0
        return covm


    def __init__(self,nv=1.0):
        """
        Initializes the :code:`Noise_Kernel` instance.

        :kwarg nv: float. Hyperparameter representing the noise level.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(nv,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(nv)
        else:
            raise ValueError('Noise hyperparameter must be a real number.')
        super(Noise_Kernel,self).__init__("n",self.__calc_covm,True,hyps)

    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        nhp = float(hyps[0])
        kcopy = Noise_Kernel(nhp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Linear_Kernel(_Kernel):
    """
    Linear Kernel: Applies linear regression :code:`ax`, can be multiplied with itself
    for higher order pure polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * pp
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = v_hyp**2.0 * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = v_hyp**2.0 * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = v_hyp**2.0 * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self,var=1.0):
        """
        Initializes the :code:`Linear_Kernel` instance.

        :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        super(Linear_Kernel,self).__init__("L",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        kcopy = Linear_Kernel(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Poly_Order_Kernel(_Kernel):
    """
    Polynomial Order Kernel: Applies linear regression :code:`ax + b`, where :code:`b != 0`,
    can be multiplied with itself for higher order polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

    :kwarg cst: float. Hyperparameter added to linear component of model, ie. :code:`b`.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        b_hyp = hyps[1]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * pp + b_hyp**2.0
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
            elif hder == 1:
                covm = b_hyp * np.ones(pp.shape)
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = v_hyp**2.0 * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = v_hyp**2.0 * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = v_hyp**2.0 * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self,var=1.0,cst=1.0):
        """
        Initializes the :code:`Poly_Order_Kernel` instance.

        :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

        :kwarg cst: float. Hyperparameter added to linear component of model, ie. :code:`b`.

        :returns: none.
        """

        hyps = np.zeros((2,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Multiplicative hyperparameter must be greater than 0.')
        if isinstance(cst,(float,int,np_itypes,np_utypes,np_ftypes)) and float(cst) > 0.0:
            hyps[1] = float(cst)
        else:
            raise ValueError('Additive hyperparameter must be greater than 0.')
        super(Poly_Order_Kernel,self).__init__("P",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        cst = float(hyps[1])
        kcopy = Poly_Order_Kernel(chp,cst)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class SE_Kernel(_Kernel):
    """
    Square Exponential Kernel: Infinitely differentiable (ie. extremely smooth) covariance function.

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        l_hyp = hyps[1]
        rr = np.abs(x1 - x2)
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * v_hyp**2.0 / np.power(l_hyp,nn)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * v_hyp / np.power(l_hyp,nn)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - jj) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * v_hyp**2.0 / np.power(l_hyp,nn + 1)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power(rr / l_hyp,nn - jj + 2)
            covm = afac * efac * sfac
        return covm


    def __init__(self,var=1.0,ls=1.0):
        """
        Initializes the :code:`SE_Kernel` instance.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter represeting variability of model in x, ie. length scale.

        :returns: none.
        """

        hyps = np.zeros((2,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        if isinstance(ls,(float,int,np_itypes,np_utypes,np_ftypes)) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Length scale hyperparameter must be greater than 0.')
        super(SE_Kernel,self).__init__("SE",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        shp = float(hyps[1])
        kcopy = SE_Kernel(chp,shp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class RQ_Kernel(_Kernel):
    """
    Rational Quadratic Kernel: Infinitely differentiable covariance function
    but provides higher tolerance for steep slopes than the squared exponential
    kernel. Mathematically equivalent to an infinite sum of squared exponential
    kernels with harmonic length scales for :code:`alpha < 20`, but becomes
    effectively identical to the squared exponential kernel as :code:`alpha`
    approaches infinity.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

    :kwarg alpha: float. Hyperparameter representing degree of length scale mixing in model.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        rq_amp = hyps[0]
        l_hyp = hyps[1]
        a_hyp = hyps[2]
        rr = np.abs(x1 - x2)
        rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * rq_amp / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii) + 1.0) / (np.power(a_hyp,nn - ii + 1) * spsp.gamma(a_hyp))
                sfac = sfac + lfac * gfac * np.power(rr / l_hyp,nn - jj + 2)
            covm = afac * efac * sfac
        elif hder == 2:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn) - 1.0)
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                pfac = (a_hyp - 2.0 * ii) / (a_hyp) * (rqt - 1.0) - float(nn - ii) / a_hyp - rqt * (np.log(rqt) + spsp.digamma(a_hyp + float(nn) - float(ii)) - spsp.digamma(a_hyp))
                sfac = sfac + cfac * gfac * pfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        return covm


    def __init__(self,amp=1.0,ls=1.0,alpha=1.0):
        """
        Initializes the :code:`RQ_Kernel` instance.

        :kwarg amp: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

        :kwarg alpha: float. Hyperparameter representing degree of length scale mixing in model.

        :returns: none.
        """

        hyps = np.zeros((3,))
        if isinstance(amp,(float,int,np_itypes,np_utypes,np_ftypes)) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Rational quadratic amplitude must be greater than 0.')
        if isinstance(ls,(float,int,np_itypes,np_utypes,np_ftypes)) and float(ls) != 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Rational quadratic hyperparameter cannot equal 0.')
        if isinstance(alpha,(float,int,np_itypes,np_utypes,np_ftypes)) and float(alpha) > 0.0:
            hyps[2] = float(alpha)
        else:
            raise ValueError('Rational quadratic alpha parameter must be greater than 0.')
        super(RQ_Kernel,self).__init__("RQ",self.__calc_covm,True,hyps)

    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        ramp = float(hyps[0])
        rhp = float(hyps[1])
        ralp = float(hyps[2])
        kcopy = RQ_Kernel(ramp,rhp,ralp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Matern_HI_Kernel(_Kernel):
    """
    Matern Kernel with Half-Integer Order Parameter: Only differentiable in
    orders less than given order parameter, :code:`nu`. Allows fit to retain
    more features at expense of volatility, but effectively becomes
    equivalent to the square exponential kernel as :code:`nu` approaches
    infinity.

    The half-integer implentation allows for use of explicit simplifications
    of the derivatives, which greatly improves its speed.

    .. note::
  
        Recommended :code:`nu = 5/2` for second order differentiability
        while retaining maximum feature representation.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

    :kwarg nu: float. Constant value setting the volatility of the model, recommended value is 2.5.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        mat_amp = hyps[0]
        mat_hyp = hyps[1]
        nu = csts[0]
        if nu < np.abs(der):
            raise ValueError('Matern nu parameter must be greater than requested derivative order.')
        pp = int(nu)
        rr = np.abs(x1 - x2)
        mht = np.sqrt(2.0 * nu) * rr / mat_hyp
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * mat_amp**2.0 * np.power(np.sqrt(2.0 * nu) / mat_hyp,nn)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * mat_amp * np.power(np.sqrt(2.0 * nu) / mat_hyp,nn)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * mat_amp**2.0 * np.power(np.sqrt(2.0 * nu),nn) / np.power(mat_hyp,nn + 1)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            ofac = np.zeros(rr.shape)
            for zz in np.arange(0,pp - nn):
                ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - nn - zz - 1))
                ofac = ofac + ffac * np.power(2.0 * mht,pp - nn - zz - 1)
            tfac = -np.power(2.0,nn + 1) * ofac
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        return covm

    def __init__(self,amp=0.1,ls=0.1,nu=2.5):
        """
        Initializes the :code:`Matern_HI_Kernel` instance.

        :kwarg amp: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

        :kwarg nu: float. Constant value setting the volatility of the model, recommended value is 2.5.

        :returns: none.
        """

        hyps = np.zeros((2,))
        csts = np.zeros((1,))
        if isinstance(amp,(float,int)) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Matern amplitude hyperparameter must be greater than 0.')
        if isinstance(ls,(float,int)) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Matern length scale hyperparameter must be greater than 0.')
        if isinstance(nu,(float,int)) and float(nu) >= 0.0:
            csts[0] = float(int(nu)) + 0.5
        else:
            raise ValueError('Matern half-integer nu constant must be greater or equal to 0.')
        super(Matern_HI_Kernel,self).__init__("MH",self.__calc_covm,True,hyps,csts)

    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        mamp = float(hyps[0])
        mhp = float(hyps[1])
        nup = float(csts[0])
        kcopy = Matern_HI_Kernel(mamp,mhp,nup)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class NN_Kernel(_Kernel):
    """
    Neural Network Style Kernel: Implements a sigmoid covariance function similar
    to a perceptron (or neuron) in a neural network, good for strong discontinuities.

    .. warning::

        Suffers from high volatility, worse than the Matern kernel. Localization of the
        kernel variation to the features in data is not yet achieved. **Strongly
        recommended NOT to use**, as Gibbs kernel provides better localized feature
        selection but is limited to a pre-defined type instead of being general.

    :kwarg nna: float. Hyperparameter representing variability of model in y.

    :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

    :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        nn_amp = hyps[0]
        nn_off = hyps[1]
        nn_hyp = hyps[2]
        rr = np.abs(x1 - x2)
        pp = x1 * x2
        nnfac = 2.0 / np.pi
        nnn = 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x1 * x2)
        nnd1 = 1.0 + 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x1**2.0)
        nnd2 = 1.0 + 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x2**2.0)
        chi = nnd1 * nnd2
        xi = chi - nnn**2.0
        covm = np.zeros(rr.shape)
        if der == 0:
            covm = nn_amp**2.0 * nnfac * np.arcsin(nnn / np.power(chi,0.5))
        elif der == 1:
            dpdx2 = x1
            dchidx2 = 4.0 * nn_hyp**2.0 * x2 * nnd1
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnm = dpdx2 * chi - dchidx2 * nnn / (4.0 * nn_hyp**2.0)
            covm = nn_amp**2.0 * nnfac * nnk * nnm
        elif der == -1:
            dpdx1 = x2
            dchidx1 = 4.0 * nn_hyp**2.0 * x1 * nnd2
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnm = dpdx1 * chi - dchidx1 * nnn / (4.0 * nn_hyp**2.0)
            covm = nn_amp**2.0 * nnfac * nnk * nnm
        elif der == 2 or der == -2:
            dpdx1 = x2
            dpdx2 = x1
            dchidx1 = 4.0 * nn_hyp**2.0 * x1 * nnd2
            dchidx2 = 4.0 * nn_hyp**2.0 * x2 * nnd1
            d2chi = 16.0 * nn_hyp**4.0 * pp
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnt1 = chi * (1.0 + (nnn / xi) * (2.0 * nn_hyp**2.0 * pp + d2chi / (8.0 * nn_hyp**2.0)))
            nnt2 = (-0.5 * chi / xi) * (dpdx2 * dchidx1 + dpdx1 * dchidx2) 
            covm = nn_amp**2.0 * nnfac * nnk * (nnt1 + nnt2)
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not implemented in '+self.name+' kernel.')
        return covm


    def __init__(self,nna=1.0,nno=1.0,nnv=1.0):
        """
        Initializes the :code:`NN_Kernel` instance.

        :kwarg nna: float. Hyperparameter representing variability of model in y.

        :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

        :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.

        :returns: none.
        """

        hyps = np.zeros((3,))
        if isinstance(nna,(float,int)) and float(nna) > 0.0:
            hyps[0] = float(nna)
        else:
            raise ValueError('Neural network amplitude must be greater than 0.')
        if isinstance(nno,(float,int)):
            hyps[1] = float(nno)
        else:
            raise ValueError('Neural network offset parameter must be a real number.')
        if isinstance(nnv,(float,int)) and float(nnv) > 0.0:
            hyps[2] = float(nnv)
        else:
            raise ValueError('Neural network hyperparameter must be a real number.')
        super(NN_Kernel,self).__init__("NN",self.__calc_covm,False,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        nnamp = float(hyps[0])
        nnop = float(hyps[1])
        nnhp = float(hyps[2])
        kcopy = NN_Kernel(nnamp,nnop,nnhp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Gibbs_Kernel(_Kernel):
    """
    Gibbs Kernel: Implements a Gibbs covariance function with variable length
    scale defined by an externally-defined warping function.

    .. note::

        The warping function is stored in the variable, :code:`self._lfunc`,
        and must be an instance of the class :code:`_WarpingFunction`. This
         was enforced to ensure functionality of hyperparameter optimization.
        Developers are **strongly recommended** to use template
        :code:`_WarpingFunction` class when implementing new warping functions
        for this package!

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg wfunc: object. Warping function, as a :code:`_WarpingFunction` instance, representing the variability of model in x as a function of x.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        l_hyp1 = self._wfunc(x1,0)
        l_hyp2 = self._wfunc(x2,0)
        rr = x1 - x2
        ll = np.power(l_hyp1,2.0) + np.power(l_hyp2,2.0)
        mm = l_hyp1 * l_hyp2
        lder = int((int(np.abs(der)) + 1) / 2)
        hdermax = self._wfunc.hyperparameters.size
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                dlh1 = self._wfunc(x1,lder,ghder)
                dlh2 = self._wfunc(x2,lder,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                covm = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
        elif der == 1:
            if hder is None:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx2 = self._wfunc(x2,lder,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx2 / (2.0 * l_hyp2) - dldx2 * dlh2 / (2.0 * np.power(l_hyp2,2.0))
                dt2 = -dlh2 * dldx2 / ll - l_hyp2 * ddldx2 / ll + l_hyp2 * dldx2 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh2 * dldx2 + 2.0 * l_hyp2 * ddldx2 - 4.0 * l_hyp2 * dldx2 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx2 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == -1:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder >= 1 and hder <= 3:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1,lder,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx1 / (2.0 * l_hyp1) - dldx1 * dlh1 / (2.0 * np.power(l_hyp1,2.0))
                dt2 = -dlh1 * dldx1 / ll - l_hyp1 * ddldx1 / ll + l_hyp1 * dldx1 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh1 * dldx1 + 2.0 * l_hyp1 * ddldx1 - 4.0 * l_hyp1 * dldx1 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx1 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == 2 or der == -2:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1,lder,ghder)
                ddldx2 = self._wfunc(x2,lder,ghder)
                ddd = ddldx1 * dldx2 + dldx1 * ddldx2
                dii = drdx1 * rr * ddldx2 / l_hyp2 - drdx1 * rr * dldx2 * dlh2 / np.power(l_hyp2,2.0) + \
                      drdx2 * rr * ddldx1 / l_hyp1 - drdx2 * rr * dldx1 * dlh1 / np.power(l_hyp1,2.0)
                djj = drdx1 * rr * ddldx2 / l_hyp2 + drdx1 * rr * dldx2 * dlh2 + \
                      drdx2 * rr * ddldx1 / l_hyp1 + drdx2 * rr * dldx1 * dlh1
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dd1 = 4.0 * dmm * np.power(rr / ll,4.0) - 16.0 * mm * dll * np.power(rr,4.0) / np.power(ll,5.0)
                dd2 = -12.0 * dmm * np.power(rr,2.0) / np.power(ll,3.0) + 36.0 * mm * dll * np.power(rr,2.0) / np.power(ll,4.0)
                dd3 = 3.0 * dmm / np.power(ll,2.0) - 6.0 * mm * dll / np.power(ll,3.0)
                dd4 = -(dll / ll + dmm / mm) * np.power(rr,2.0) / (ll * mm)
                dd5 = dmm / (4.0 * np.power(mm,2.0))
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                ddt = ddd * (d1 + d2 + d3 + d4 + d5) + dd * (dd1 + dd2 + dd3 + dd4 + dd5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                djt1 = 6.0 * djj / np.power(ll,2.0) - 12.0 * jj * dll / np.power(ll,3.0)
                djt2 = -4.0 * djj * np.power(rr,2.0) / np.power(ll,3.0) + 12.0 * jj * dll * np.power(rr,2.0) / np.power(ll,4.0)
                djt3 = dii / ll - ii * dll / np.power(ll,2.0)
                djt = djt1 + djt2 + djt3
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                drt = -2.0 * drdx1 * drdx2 * (4.0 * np.power(rr,2.0) / np.power(ll,3.0) - 1.0 / np.power(ll,2.0))
                covm = dkfac * (dt + jt + rt) + kfac * (ddt + djt + drt)
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not implemented in '+self.name+' kernel.')
        return covm


    @property
    def wfuncname(self):
        """
        Returns the codename of the stored :code:`_WarpingFunction` instance.

        :returns: str. Codename of the stored :code:`_WarpingFunction` instance.
        """

        # Ensure reconstruction failure if warping function is not properly defined
        wfname = "?"
        if isinstance(self._wfunc,_WarpingFunction):
            wfname = self._wfunc.name 
        else:
            warnings.warn('Gibbs_Kernel warping function is not a valid WarpingFunction object.')
        return wfname


    def evaluate_wfunc(self,xx,der=0,hder=None):
        """
        Evaluates the stored warping function at the specified values.

        :arg xx: array. Vector of x-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of x derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values, under the given derivative settings. Has the same dimensions as :code:`xx`.
        """

        # Prevent catastrophic failure if warping function is not properly defined
        lsf = None
        if isinstance(self._wfunc,_WarpingFunction):
            lsf = self._wfunc(xx,der,hder)
        else:
            warnings.warn('Gibbs_Kernel warping function is not a valid WarpingFunction object.')
        return lsf


    def __init__(self,var=1.0,wfunc=None):
        """
        Initialize the :code:`Gibbs_Kernel` instance.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg wfunc: object. Warping function, as a :code:`_WarpingFunction` instance, representing the variability of model in x as a function of x.

        :returns: none.
        """

        self._wfunc = None
        if isinstance(wfunc,_WarpingFunction):
            self._wfunc = copy.copy(wfunc)
        elif wfunc is None:
            self._wfunc = Constant_WarpingFunction(1.0e0)

        hyps = np.zeros((1,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(var)
        else:
            raise ValueError('Amplitude hyperparameter must be a real number.')
        super(Gibbs_Kernel,self).__init__("G",self.__calc_covm,True,hyps)


    @property
    def name(self):

        name = super(Gibbs_Kernel,self).name
        if isinstance(self._wfunc,_WarpingFunction):
            name = name + "w" + self._wfunc.name
        return name


    @property
    def hyperparameters(self):

        val = super(Gibbs_Kernel,self).hyperparameters
        if isinstance(self._wfunc,_WarpingFunction):
            val = np.hstack((val,self._wfunc.hyperparameters))
        return val


    @property
    def constants(self):

        val = super(Gibbs_Kernel,self).constants
        if isinstance(self._wfunc,_WarpingFunction):
            val = np.hstack((val,self._wfunc.constants))
        return val


    @property
    def bounds(self):

        val = super(Gibbs_Kernel,self).bounds
        if isinstance(self._wfunc,_WarpingFunction):
            wval = self._wfunc.bounds
            if wval is not None:
                val = np.hstack((val,wval)) if val is not None else wval
        return val


    @hyperparameters.setter
    def hyperparameters(self,theta):
        """
        Set the hyperparameters stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        """

        userhyps = None
        if isinstance(theta,(list,tuple,array)):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError('%s Kernel hyperparameters must be given as an array-like object.' % (self._fname))
        if super(Gibbs_Kernel,self).hyperparameters.size > 0:
            super(Gibbs_Kernel,self.__class__).hyperparameters.__set__(self,userhyps)
        if isinstance(self._wfunc,_WarpingFunction):
            nhyps = super(Gibbs_Kernel,self).hyperparameters.size
            if nhyps < userhyps.size:
                self._wfunc.hyperparameters = userhyps[nhyps:]
        else:
            warnings.warn('%s warping function is not a valid WarpingFunction instance.' % (type(self).__name__))


    @constants.setter
    def constants(self,consts):
        """
        Set the constants stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        """

        usercsts = None
        if isinstance(consts,(list,tuple,array)):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError('%s Kernel constants must be given as an array-like object.' % (self._fname))
        if super(Gibbs_Kernel,self).constants.size > 0:
            super(Gibbs_Kernel,self.__class__).constants.__set__(self,usercsts)
        if isinstance(self._wfunc,_WarpingFunction):
            ncsts = super(Gibbs_Kernel,self).constants.size
            if ncsts < usercsts.size:
                self._wfunc.constants = usercsts[ncsts:]
        else:
            warnings.warn('%s warping function is not a valid WarpingFunction object.' % (type(self).__name__))


    @bounds.setter
    def bounds(self,bounds):
        """
        Set the hyperparameter bounds stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        """

        userbnds = None
        if isinstance(bounds,(list,tuple,array)):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError('%s Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        if userbnds.shape[0] != 2:
            raise TypeError('%s Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.' % (self._fname))
        super(Gibbs_Kernel,self.__class__).bounds.__set__(self,userbnds)
        if isinstance(self._wfunc,_WarpingFunction):
            wbnds = super(Gibbs_Kernel,self).bounds
            nbnds = wbnds.shape[1] if wbnds is not None else 0
            if nbnds < userbnds.shape[1]:
                self._wfunc.bounds = userbnds[:,nbnds:]
        else:
            warnings.warn('%s warping function is not a valid WarpingFunction object.' % (type(self).__name__))


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        wfunc = copy.copy(self._wfunc)
        kcopy = Gibbs_Kernel(chp,wfunc)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Constant_WarpingFunction(_WarpingFunction):
    """
    Constant Warping Function for Gibbs Kernel: effectively reduces Gibbs kernel to squared exponential kernel.
    
    :kwarg cv: float. Hyperparameter representing constant value which the warping function always evalutates to.
    """

    def __calc_warp(self,zz,der=0,hder=None):
        """
        Implementation-specific warping function.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        c_hyp = hyps[0]
        warp = np.zeros(zz.shape)
        if der == 0:
            if hder is None:
                warp = c_hyp * np.ones(zz.shape)
            elif hder == 0:
                warp = np.ones(zz.shape)
        return warp


    def __init__(self,cv=1.0):
        """
        Initializes the :code:`Constant_WarpingFunction` instance.

        :kwarg cv: float. Hyperparameter representing constant value which warping function always evaluates to.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(cv,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super(Constant_WarpingFunction,self).__init__("C",self.__calc_warp,True,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        kcopy = Constant_WarpingFunction(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class IG_WarpingFunction(_WarpingFunction):
    """
    Inverse Gaussian Warping Function for Gibbs Kernel: localized variation of length-scale with variation limit.

    :kwarg lb: float. Hyperparameter representing base length scale.

    :kwarg gh: float. Hyperparameter representing height of Gaussian envelope adjusting the length scale.

    :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

    :kwarg gm: float. Constant indicating location of peak of Gaussian envelope adjusting the length scale.

    :kwarg mf: float. Constant indicating upper limit for height-to-base length scale ratio, to improve stability.
    """

    def __calc_warp(self,zz,der=0,hder=None):
        """
        Implementation-specific warping function.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        """

        hyps = self.hyperparameters
        csts = self.constants
        base = hyps[0]
        amp = hyps[1]
        sig = hyps[2]
        mu = csts[0]
        maxfrac = csts[1]
        nn = int(np.abs(der))
        hh = amp if amp < (maxfrac * base) else maxfrac * base
        warp = np.ones(zz.shape) * base
        if hder is None:
            afac = -hh * np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig,nn - jj)
            warp = base + afac * sfac if der == 0 else afac * sfac
        elif hder == 0:
            warp = np.ones(zz.shape) if der == 0 else np.zeros(zz.shape)
        elif hder == 1:
            afac = -np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig,nn - jj)
            warp = afac * sfac
        elif hder == 2:
            afac = -hh * np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn + 1)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power((zz - mu) / sig,nn - jj + 2)
            warp = afac * sfac
        return warp


    def __init__(self,lb=1.0,gh=0.5,gs=1.0,gm=0.0,mf=0.6):
        """
        Initializes the :code:`IG_WarpingFunction` instance.

        :kwarg lb: float. Hyperparameter representing base length scale.

        :kwarg gh: float. Hyperparameter representing height of Gaussian envelope adjusting the length scale.

        :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

        :kwarg gm: float. Constant indicating location of peak of Gaussian envelope adjusting the length scale.

        :kwarg mf: float. Constant indicating upper limit for height-to-base length scale ratio, to improve stability.

        :returns: none.
        """

        hyps = np.zeros((3,))
        csts = np.zeros((2,))
        if isinstance(lb,(float,int,np_itypes,np_utypes,np_ftypes)) and float(lb) > 0.0:
            hyps[0] = float(lb)
        else:
            raise ValueError('Length scale function base hyperparameter must be greater than 0.')
        if isinstance(gh,(float,int,np_itypes,np_utypes,np_ftypes)) and float(gh) > 0.0:
            hyps[1] = float(gh)
        else:
            raise ValueError('Length scale function minimum hyperparameter must be greater than 0.')
        if isinstance(gs,(float,int,np_itypes,np_utypes,np_ftypes)) and float(gs) > 0.0:
            hyps[2] = float(gs)
        else:
            raise ValueError('Length scale function sigma hyperparameter must be greater than 0.')
        if isinstance(gm,(float,int,np_itypes,np_utypes,np_ftypes)):
            csts[0] = float(gm)
        else:
            raise ValueError('Length scale function mu constant must be a real number.')
        if isinstance(mf,(float,int,np_itypes,np_utypes,np_ftypes)) and float(mf) < 1.0:
            csts[1] = float(mf)
        else:
            raise ValueError('Length scale function minimum-to-base ratio limit must be less than 1.')
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = float(csts[1] * hyps[0])
        super(IG_WarpingFunction,self).__init__("IG",self.__calc_warp,True,hyps,csts)


    @property
    def hyperparameters(self):

        return super(IG_WarpingFunction,self).hyperparameters


    @property
    def constants(self):

        return super(IG_WarpingFunction,self).constants


    @property
    def bounds(self):

        return super(IG_WarpingFunction,self).bounds


    @hyperparameters.setter
    def hyperparameters(self,theta):
        """
        Set the hyperparameters stored in the :code:`_WarpingFunction` object. Specific implementation due to maximum fraction limit.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        super(IG_WarpingFunction,self.__class__).hyperparameters.__set__(self,theta)
        hyps = self.hyperparameters
        csts = self.constants
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = csts[1] * hyps[0]
            super(IG_WarpingFunction,self.__class__).hyperparameters.__set__(self,hyps)


    @constants.setter
    def constants(self,consts):
        """
        Set the constants stored in the :code:`_WarpingFunction` object. Specific implementation due to maximum fraction limit.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        super(IG_WarpingFunction,self.__class__).constants.__set__(self,consts)
        hyps = self.hyperparameters
        csts = self.constants
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = csts[1] * hyps[0]
            super(IG_WarpingFunction,self.__class__).hyperparameters.__set__(self,hyps)


    @bounds.setter
    def bounds(self,bounds):
        """
        Set the hyperparameter bounds stored in the :code:`_WarpingFunction` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        """

        super(IG_WarpingFunction,self.__class__).bounds.__set__(self,bounds)
        if self._force_bounds:
            hyps = self.hyperparameters
            csts = self.constants
            if hyps[1] > (csts[1] * hyps[0]):
                hyps[1] = csts[1] * hyps[0]
                super(IG_WarpingFunction,self.__class__).hyperparameters.__set__(self,hyps)


    def __copy__(self):
        """
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        """

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        lbhp = float(hyps[0])
        ghhp = float(hyps[1])
        gshp = float(hyps[2])
        gmc = float(csts[0])
        lrc = float(csts[1])
        kcopy = IG_WarpingFunction(lbhp,ghhp,gshp,gmc,lrc)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class GaussianProcessRegression1D(object):
    """
    Class containing variable containers, get/set functions, and fitting functions required to
    perform Gaussian process regressions on 1-dimensional data.

    .. note::

        This implementation requires the specific implementation of the :code:`_Kernel`
        template class provided within the same package!
    """

    def __init__(self):
        """
        Defines the input and output containers used within the class, but they still requires instantiation.
        """

        self._kk = None
        self._kb = None
        self._lp = 1.0
        self._xx = None
        self._xe = None
        self._yy = None
        self._ye = None
        self._dxx = None
        self._dyy = None
        self._dye = None
        self._eps = None
        self._opm = 'grad'
        self._opp = np.array([1.0e-5])
        self._dh = 1.0e-2
        self._lb = None
        self._ub = None
        self._cn = None
        self._ekk = None
        self._ekb = None
        self._elp = 6.0
        self._enr = None
        self._eeps = None
        self._eopm = 'grad'
        self._eopp = np.array([1.0e-5])
        self._edh = 1.0e-2
        self._ikk = None
        self._imax = 500
        self._xF = None
        self._estF = None
        self._barF = None
        self._varF = None
        self._dbarF = None
        self._dvarF = None
        self._lml = None
        self._nulllml = None
        self._barE = None
        self._varE = None
        self._dbarE = None
        self._dvarE = None
        self._varN = None
        self._dvarN = None
        self._gpxe = None
        self._gpye = None
        self._egpye = None
        self._nikk = None
        self._niekk = None
        self._fwarn = False
        self._opopts = ['grad','mom','nag','adagrad','adadelta','adam','adamax','nadam']


    def __eq__(self,other):
        """
        Custom equality operator, only compares input data due to statistical
        variance of outputs.

        :arg other: object. Another :code:`GaussianProcessRegression1D` object.

        :returns: bool. Indicates whether the two objects have identical inputs.
        """

        status = False
        if isinstance(other,GaussianProcessRegression1D):
            skk = self._kk.name == other._kk.name if self._kk is not None and other._kk is not None else self._kk == other._kk
            skb = np.all(np.isclose(self._kb,other._kb)) if self._kb is not None and other._kb is not None else np.all(np.atleast_1d(self._kb == other._kb))
            seps = np.isclose(self._eps,other._eps) if self._eps is not None and other._eps is not None else self._eps == other._eps
            sekk = self._ekk.name == other._ekk.name if self._ekk is not None and other._ekk is not None else self._ekk == other._ekk
            sekb = np.all(np.isclose(self._ekb,other._ekb)) if self._ekb is not None and other._ekb is not None else np.all(np.atleast_1d(self._ekb == other._ekb))
            seeps = np.isclose(self._eeps,other._eeps) if self._eeps is not None and other._eeps is not None else self._eeps == other._eeps
            sxx = np.all(np.isclose(self._xx,other._xx)) if self._xx is not None and other._xx is not None else np.all(np.atleast_1d(self._xx == other._xx))
            sxe = np.all(np.isclose(self._xe,other._xe)) if self._xe is not None and other._xe is not None else np.all(np.atleast_1d(self._xe == other._xe))
            syy = np.all(np.isclose(self._yy,other._yy)) if self._yy is not None and other._yy is not None else np.all(np.atleast_1d(self._yy == other._yy))
            sye = np.all(np.isclose(self._ye,other._ye)) if self._ye is not None and other._ye is not None else np.all(np.atleast_1d(self._ye == other._ye))
            sdxx = np.all(np.isclose(self._dxx,other._dxx)) if self._dxx is not None and other._dxx is not None else np.all(np.atleast_1d(self._dxx == other._dxx))
            sdyy = np.all(np.isclose(self._dyy,other._dyy)) if self._dyy is not None and other._dyy is not None else np.all(np.atleast_1d(self._dyy == other._dyy))
            sdye = np.all(np.isclose(self._dye,other._dye)) if self._dye is not None and other._dye is not None else np.all(np.atleast_1d(self._dye == other._dye))
            slb = np.isclose(self._lb,other._lb) if self._lb is not None and other._lb is not None else self._lb == other._lb
            sub = np.isclose(self._ub,other._ub) if self._ub is not None and other._ub is not None else self._ub == other._ub
            scn = np.isclose(self._cn,other._cn) if self._cn is not None and other._cn is not None else self._cn == other._cn
            #print(skk,skb,seps,sekk,sekb,seeps,sxx,sxe,sye,sdxx,sdyy,sdye,slb,sub,scn)
            status = skk and skb and seps and sekk and sekb and seeps and \
                     sxx and sxe and syy and sye and sdxx and sdyy and sdye and \
                     np.isclose(self._lp,other._lp) and np.isclose(self._elp,other._elp) and \
                     slb and sub and scn and \
                     (self._opm == other._opm and np.all(np.isclose(self._opp,other._opp))) and \
                     (self._eopm == other._eopm and np.all(np.isclose(self._eopp,other._eopp)))
            
        return status


    def __ne__(self,other):
        """
        Custom inequality operator, only compares input data due to statistical
        variance of outputs.

        :arg other: object. Another :code:`GaussianProcessRegression1D` object.

        :returns: bool. Indicates whether the two objects do not have identical inputs.
        """

        return not self.__eq__(other)


    def set_kernel(self,kernel=None,kbounds=None,regpar=None):
        """
        Specify the kernel that the Gaussian process regression will be performed with.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility. (optional)

        :returns: none.
        """

        if isinstance(kernel,_Kernel):
            self._kk = copy.copy(kernel)
            self._ikk = copy.copy(self._kk)
        if isinstance(self._kk,_Kernel):
            kh = np.log10(self._kk.hyperparameters)
            if isinstance(kbounds,(list,tuple,np.ndarray)):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb.T) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._kb = kb
                self._kk.bounds = np.power(10.0,self._kb)
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            self._lp = float(regpar)


    def set_raw_data(self,xdata=None,ydata=None,xerr=None,yerr=None,dxdata=None,dydata=None,dyerr=None):
        """
        Specify the raw data that the Gaussian process regression will be performed on.
        Performs some consistency checks between the input raw data to ensure validity.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg xerr: array. Vector of x-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. Vector of x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. Vector of dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. Vector of dy/dx-errors of derivative data points to be included in fit. (optional)

        :returns: none.
        """

        altered = False
        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            self._xx = np.array(xdata).flatten()
            altered = True
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            self._xx = xdata.flatten()
            altered = True
        if isinstance(xerr,(list,tuple)) and len(xerr) > 0:
            self._xe = np.array(xerr).flatten()
            altered = True
        elif isinstance(xerr,np.ndarray) and xerr.size > 0:
            self._xe = xerr.flatten()
            altered = True
        elif isinstance(xerr,str):
            self._xe = None
            altered = True
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            self._yy = np.array(ydata).flatten()
            altered = True
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            self._yy = ydata.flatten()
            altered = True
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            self._ye = np.array(yerr).flatten()
            altered = True
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            self._ye = yerr.flatten()
            altered = True
        elif isinstance(yerr,str):
            self._ye = None
            altered = True
        if isinstance(dxdata,(list,tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dxx = temp.flatten()
            altered = True
        elif isinstance(dxdata,np.ndarray) and dxdata.size > 0:
            self._dxx = dxdata.flatten()
            altered = True
        elif isinstance(dxdata,str):
            self._dxx = None
            altered = True
        if isinstance(dydata,(list,tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dyy = temp.flatten()
            altered = True
        elif isinstance(dydata,np.ndarray) and dydata.size > 0:
            self._dyy = dydata.flatten()
            altered = True
        elif isinstance(dydata,str):
            self._dyy = None
            altered = True
        if isinstance(dyerr,(list,tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dye = temp.flatten()
            altered = True
        elif isinstance(dyerr,np.ndarray) and dyerr.size > 0:
            self._dye = dyerr.flatten()
            altered = True
        elif isinstance(dyerr,str):
            self._dye = None
            altered = True
        if altered:
            self._gpxe = None
            self._gpye = None
            self._egpye = None
            self._nikk = None
            self._niekk = None


    def set_conditioner(self,condnum=None,lbound=None,ubound=None):
        """
        Specify the parameters to ensure the condition number of the matrix is good,
        as well as set upper and lower bounds for the input data to be included.

        :kwarg condnum: float. Minimum allowable delta-x for input data before applying Gaussian blending to data points.

        :kwarg lbound: float. Minimum allowable y-value for input data, values below are omitted from fit procedure. (optional)

        :kwarg ubound: float. Maximum allowable y-value for input data, values above are omitted from fit procedure. (optional)

        :returns: none.
        """

        if isinstance(condnum,(float,int,np_itypes,np_utypes,np_ftypes)) and float(condnum) > 0.0:
            self._cn = float(condnum)
        elif isinstance(condnum,(float,int,np_itypes,np_utypes,np_ftypes)) and float(condnum) <= 0.0:
            self._cn = None
        elif isinstance(condnum,str):
            self._cn = None
        if isinstance(lbound,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._lb = float(lbound)
        elif isinstance(lbound,str):
            self._lb = None
        if isinstance(ubound,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._ub = float(ubound)
        elif isinstance(ubound,str):
            self._ub = None


    def set_error_kernel(self,kernel=None,kbounds=None,regpar=None,nrestarts=None):
        """
        Specify the kernel that the Gaussian process regression on the error function
        will be performed with.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the error data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility. (optional)

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within :code:`kbounds`. (optional)

        :returns: none.
        """

        altered = False
        if isinstance(kernel,_Kernel):
            self._ekk = copy.copy(kernel)
            altered = True
        if isinstance(self._ekk,_Kernel):
            kh = np.log10(self._ekk.hyperparameters)
            if isinstance(kbounds,(list,tuple,np.ndarray)):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb.T) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._ekb = kb
                self._ekk.bounds = np.power(10.0,self._ekb)
                altered = True
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            self._elp = float(regpar)
            altered = True
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._enr = int(nrestarts) if int(nrestarts) > 0 else 0
        if altered:
            self._gpxe = None
            self._gpye = None
            self._egpye = None
            self._nikk = None
            self._niekk = None


    def set_search_parameters(self,epsilon=None,method=None,spars=None,sdiff=None,maxiter=None):
        """
        Specify the search parameters that the Gaussian process regression will use.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include:
                       ['grad', 'mom', 'nag', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.
	              **Only** used if analytical implementation of kernel derivative is not present! (optional)

        :kwarg maxiter: int. Maximum number of iterations for hyperparameter optimization algorithm, default is 500. (optional)

        :returns: none.
        """

        midx = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            self._eps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            self._eps = None
        elif isinstance(epsilon,str):
            self._eps = None
        if isinstance(method,str):
            mstr = method.lower()
            if mstr in self._opopts:
                midx = self._opopts.index(mstr)
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) >= 0 and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                self._opm = self._opopts[1]
                opp = np.array([1.0e-4,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 2:
                self._opm = self._opopts[2]
                opp = np.array([1.0e-4,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 3:
                self._opm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 4:
                self._opm = self._opopts[4]
                opp = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 5:
                self._opm = self._opopts[5]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 6:
                self._opm = self._opopts[6]
                opp = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 7:
                self._opm = self._opopts[7]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            else:
                self._opm = self._opopts[0]
                opp = np.array([1.0e-4]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < self._opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._opp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < self._opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._opp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            self._dh = float(sdiff)
        if isinstance(maxiter,(float,int,np_itypes,np_utypes,np_ftypes)) and int(maxiter) > 0:
            self._imax = int(maxiter) if int(maxiter) > 50 else 50


    def set_error_search_parameters(self,epsilon=None,method=None,spars=None,sdiff=None):
        """
        Specify the search parameters that the Gaussian process regression will use for the error function.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include:
                       ['grad', 'mom', 'nag', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.
	              **Only** used if analytical implementation of kernel derivative is not present! (optional)

        :returns: none.
        """

        emidx = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            self._eeps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            self._eeps = None
        elif isinstance(epsilon,str):
            self._eeps = None
        if isinstance(method,str):
            mstr = method.lower()
            if mstr in self._opopts:
                emidx = self._opopts.index(mstr)
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) >= 0 and int(method) < len(self._opopts):
            emidx = int(method)
        if emidx is not None:
            if emidx == 1:
                self._eopm = self._opopts[1]
                opp = np.array([1.0e-4,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 2:
                self._eopm = self._opopts[2]
                opp = np.array([1.0e-4,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 3:
                self._eopm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 4:
                self._eopm = self._opopts[4]
                opp = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 5:
                self._eopm = self._opopts[5]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 6:
                self._eopm = self._opopts[6]
                opp = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 7:
                self._eopm = self._opopts[7]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            else:
                self._eopm = self._opopts[0]
                opp = np.array([1.0e-4]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < self._eopp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._eopp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < self._eopp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._eopp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            self._edh = float(sdiff)


    def set_warning_flag(self,flag=True):
        """
        Specify the printing of runtime warnings within the
        hyperparameter optimization routine. The warnings are
        disabled by default but calling this function will
        enable them by default.

        :kwarg flag: bool. Flag to toggle display of warnings.

        :returns: none.
        """

        self._fwarn = True if flag else False


    def reset_error_kernel(self):
        """
        Resets error kernel and associated settings to an empty
        state. Primarily used for setting up identical objects
        for comparison and testing purposes.

        :returns: none.
        """

        self._ekk = None
        self._ekb = None
        self._elp = 6.0
        self._enr = None
        self._eeps = None
        self._eopm = 'grad'
        self._eopp = np.array([1.0e-5])
        self._edh = 1.0e-2
        self._gpxe = None
        self._gpye = None
        self._egpye = None
        self._nikk = None


    def get_raw_data(self):
        """
        Returns the input raw data passed in latest :code:`set_raw_data()` call,
        without any internal processing.

        :returns: (array, array, array, array, array, array, array).
            Vectors in order of x-values, y-values, x-errors, y-errors, derivative x-values, dy/dx-values, dy/dx-errors.
        """

        rxx = copy.deepcopy(self._xx)
        ryy = copy.deepcopy(self._yy)
        rxe = copy.deepcopy(self._xe)
        rye = copy.deepcopy(self._ye)
        rdxx = copy.deepcopy(self._dxx)
        rdyy = copy.deepcopy(self._dyy)
        rdye = copy.deepcopy(self._dye)
        return (rxx,ryy,rxe,rye,rdxx,rdyy,rdye)


    def get_processed_data(self):
        """
        Returns the input data passed into the latest :code:`GPRFit()` call,
        including all internal processing performed by that call.

        .. note::

            If :code:`GPRFit()` was executed with :code:`nigp_flag = True`, then
            the raw x-error data is folded into the y-error. As such, this
            function only returns y-errors.

        :returns: (array, array, array, array, array, array, array).
            Vectors in order of x-values, y-values, y-errors, derivative x-values, dy/dx-values, dy/dx-errors.
        """

        pxx = copy.deepcopy(self._xx)
        pyy = copy.deepcopy(self._yy)
        pye = copy.deepcopy(self._ye)
        if isinstance(pxx,np.ndarray) and isinstance(pyy,np.ndarray):
            rxe = self._xe if self._xe is not None else np.zeros(pxx.shape)
            rye = self._ye if self._gpye is None else self._gpye
            if rye is None:
                rye = np.zeros(pyy.shape)
            lb = -1.0e50 if self._lb is None else self._lb
            ub = 1.0e50 if self._ub is None else self._ub
            cn = 5.0e-3 if self._cn is None else self._cn
            (pxx,pxe,pyy,pye,nn) = self._condition_data(self._xx,rxe,self._yy,rye,lb,ub,cn)
        # Actually these should be conditioned as well (for next version?)
        dxx = copy.deepcopy(self._dxx)
        dyy = copy.deepcopy(self._dyy)
        dye = copy.deepcopy(self._dye)
        return (pxx,pyy,pye,dxx,dyy,dye)


    def get_gp_x(self):
        """
        Returns the x-values used in the latest :code:`GPRFit()` call.

        :returns: array. Vector of x-values corresponding to predicted y-values.
        """

        return copy.deepcopy(self._xF)


    def get_gp_regpar(self):
        """
        Returns the regularization parameter value used in the latest :code:`GPRFit()` call.

        :returns: float. Regularization parameter value used in cost function evaluation.
        """

        return self._lp


    def get_gp_error_regpar(self):
        """
        Returns the regularization parameter value used for error function fitting in the latest :code:`GPRFit()` call.

        :returns: float. Regularization parameter value used in cost function evaluation for error function fitting.
        """

        return self._elp


    def get_gp_mean(self):
        """
        Returns the y-values computed in the latest :code:`GPRFit()` call.

        :returns: array. Vector of predicted y-values from fit.
        """

        return copy.deepcopy(self._barF)


    # TODO: Place process noise fraction on GPRFit() level and remove the argument from these functions, currently introduces inconsistencies in statistics
    def get_gp_variance(self,noise_flag=True,noise_mult=None):
        """
        Returns the full covariance matrix of the y-values computed in the latest
        :code:`GPRFit()` call.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg noise_mult: float. Noise term multiplier to introduce known bias or covariance in data, must be greater than or equal to zero. (optional)

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        """

        varF = copy.deepcopy(self._varF)
        if varF is not None and self._varN is not None and noise_flag:
            nfac = float(noise_mult) ** 2.0 if isinstance(noise_mult,(float,int,np_itypes,np_utypes,np_ftypes)) and float(noise_mult) >= 0.0 else 1.0
            varF = varF + nfac * self._varN
        return varF


    def get_gp_std(self,noise_flag=True,noise_mult=None):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma error of fit.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors. (optional)

        :kwarg noise_mult: float. Noise term multiplier to introduce known bias or covariance in data, must be greater than or equal to zero. (optional)

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        """

        sigF = None
        varF = self.get_gp_variance(noise_flag=noise_flag,noise_mult=noise_mult)
        if varF is not None:
            sigF = np.sqrt(np.diag(varF))
        return sigF


    def get_gp_drv_mean(self):
        """
        Returns the dy/dx-values computed in the latest :code:`GPRFit()` call.

        :returns: array. Vector of predicted dy/dx-values from fit, if requested in fit call.
        """

        return copy.deepcopy(self._dbarF)


    def get_gp_drv_variance(self,noise_flag=True,process_noise_fraction=None):
        """
        Returns the full covariance matrix of the dy/dx-values computed in the latest
        :code:`GPRFit()` call.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: array. 2D meshgrid array containing full covariance matrix for predicted dy/dx-values from fit, if requested in fit call.
        """

        dvarF = copy.deepcopy(self._dvarF)
        if dvarF is not None:
            dvar_numer = self.get_gp_variance(noise_flag=noise_flag,noise_mult=process_noise_fraction)
            dvar_denom = self.get_gp_variance(noise_flag=False)
            dvar_denom[dvar_denom == 0.0] = 1.0
            dvarmod = dvar_numer / dvar_denom
            if self._dvarN is not None and noise_flag:
                nfac = float(process_noise_fraction) ** 2.0 if isinstance(process_noise_fraction,(float,int,np_itypes,np_utypes,np_ftypes)) and float(process_noise_fraction) >= 0.0 and float(process_noise_fraction) <= 1.0 else 1.0
                dvarF = dvarmod * dvarF + nfac * self._dvarN
            else:
                dvarF = dvarmod * dvarF
        return dvarF


    def get_gp_drv_std(self,noise_flag=True,process_noise_fraction=None):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the 
        dy/dx-values computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma
        error of fit.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: array. 1D array containing 1 sigma errors of predicted dy/dx-values from fit, if requested in fit call.
        """

        dsigF = None
        dvarF = self.get_gp_drv_variance(noise_flag=noise_flag,process_noise_fraction=process_noise_fraction)
        if dvarF is not None:
            dsigF = np.sqrt(np.diag(dvarF))
        return dsigF


    def get_gp_results(self,rtn_cov=False,noise_flag=True,process_noise_fraction=None):
        """
        Returns all common predicted values computed in the latest :code:`GPRFit()` call.

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead the 1 sigma errors. (optional)

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variances or errors. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: (array, array, array, array).
            Vectors in order of y-values, y-errors, dy/dx-values, dy/dx-errors.
        """

        ra = self.get_gp_mean()
        rb = self.get_gp_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_std(noise_flag=noise_flag)
        rc = self.get_gp_drv_mean()
        rd = self.get_gp_drv_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_drv_std(noise_flag=noise_flag,process_noise_fraction=process_noise_fraction)
        return (ra,rb,rc,rd)


    def get_gp_lml(self):
        """
        Returns the log-marginal-likelihood of the latest :code:`GPRFit()` call.

        :returns: float. Log-marginal-likelihood value from fit.
        """

        return self._lml


    def get_gp_null_lml(self):
        """
        Returns the log-marginal-likelihood for the null hypothesis, calculated by the latest :code:`GPRFit()` call.
        This value can be used to normalize the log-marginal-likelihood of the fit for a generalized goodness-of-fit metric.

        :returns: float. Log-marginal-likelihood value of null hypothesis.
        """

        return self._nulllml


    def get_gp_adjusted_r2(self):
        """
        Calculates the adjusted R-squared (coefficient of determination) using the results of the latest :code:`GPRFit()`
        call.

        :returns: float. Adjusted R-squared value.
        """

        adjr2 = None
        if self._xF is not None and self._estF is not None:
            myy = np.nanmean(self._yy)
            sstot = np.sum(np.power(self._yy - myy,2.0))
            ssres = np.sum(np.power(self._yy - self._estF,2.0))
            kpars = np.hstack((self._kk.hyperparameters,self._kk.constants))
            adjr2 = 1.0 - (ssres / sstot) * (self._xx.size - 1.0) / (self._xx.size - kpars.size - 1.0)
        return adjr2


    def get_gp_generalized_r2(self):
        """
        Calculates the Cox and Snell pseudo R-squared (coefficient of determination) using the results of the latest
        :code:`GPRFit()` call.

        .. note:: This particular metric is for logistic regression and may not be fully applicable to generalized polynomial
                  regression. However, they are related here through the use of maximum likelihood optimization. Use with
                  extreme caution!!!

        :returns: float. Generalized pseudo R-squared value based on Cox and Snell methodology.
        """

        genr2 = None
        if self._xF is not None:
            genr2 = 1.0 - np.exp(2.0 * (self._nulllml - self._lml) / self._xx.size)
        return genr2


    def get_gp_input_kernel(self):
        """
        Returns the original input kernel, with settings retained from before the
        hyperparameter optimization step.

        :returns: object. The original input :code:`_Kernel` instance, saved from the latest :code:`set_kernel()` call.
        """

        return self._ikk


    def get_gp_kernel(self):
        """
        Returns the optimized kernel determined in the latest :code:`GPRFit()` call.

        :returns: object. The :code:`_Kernel` instance from the latest :code:`GPRFit()` call, including optimized hyperparameters if fit was performed.
        """

        return self._kk


    def get_gp_kernel_details(self):
        """
        Returns the data needed to save the optimized kernel determined in the latest :code:`GPRFit()` call.

        :returns: (str, array, float).
            Kernel codename, vector of kernel hyperparameters and constants, regularization parameter.
        """

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._kk,_Kernel):
            kname = self._kk.name
            kpars = np.hstack((self._kk.hyperparameters,self._kk.constants))
            krpar = self._lp
        return (kname,kpars,krpar)


    def get_gp_error_kernel(self):
        """
        Returns the optimized error kernel determined in the latest :code:`GPRFit()` call.

        :returns: object. The error :code:`_Kernel` instance from the latest :code:`GPRFit()` call, including optimized hyperparameters if fit was performed.
        """

        return self._ekk


    def get_gp_error_kernel_details(self):
        """
        Returns the data needed to save the optimized error kernel determined in the latest :code:`GPRFit()` call.

        :returns: (str, array).
            Kernel codename, vector of kernel hyperparameters and constants, regularization parameter.
        """

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._ekk,_Kernel):
            kname = self._ekk.name
            kpars = np.hstack((self._ekk.hyperparameters,self._ekk.constants))
            krpar = self._elp
        return (kname,kpars,krpar)


    def get_error_gp_mean(self):
        """
        Returns the fitted y-errors computed in the latest :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. Vector of predicted y-values from fit.
        """

        return copy.deepcopy(self._barE)


    def get_error_gp_variance(self):
        """
        Returns the full covariance matrix of the fitted y-errors computed in the latest
        :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        """

        return copy.deepcopy(self._varE)


    def get_error_gp_std(self):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma error of fit.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        """

        sigE = None
        varE = self.get_error_gp_variance()
        if varE is not None:
            sigE = np.sqrt(np.diag(varE))
        return sigE


    def eval_error_function(self,xnew,enforce_positive=True):
        """
        Returns the error values used in heteroscedastic GPR, evaluated at the input x-values,
        using the error kernel determined in the latest :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :arg xnew: array. Vector of x-values at which the predicted error function should be evaluated at.

        :kwarg enforce_positive: bool. Returns of absolute values of the error function if :code:`True`.

        :returns: array. Vecotr of predicted y-errors from the fit using the error kernel.
        """

        xn = None
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        barE = None
        if xn is not None and self._gpye is not None and self._egpye is not None:
            barE = itemgetter(0)(self.__basic_fit(xn,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,epsilon='None'))
            if enforce_positive:
                barE = np.abs(barE)
        return barE


    def _gp_base_alg(self,xn,kk,lp,xx,yy,ye,dxx,dyy,dye,dd):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for 1-dimensional Gaussian process regression, with no
        idiot-proofing and no pre- or post-processing.

        .. note::

            It is **strongly recommended** that :code:`kk` be a :code:`_Kernel` instance as
            specified in this package but, within this function, it can essentially be any
            callable object which accepts the arguments :code:`(x1,x2,dd)`.

        :arg xn: array. Vector of x-values at which the fit will be evaluated.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given in 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg dd: int. Derivative order of output prediction.

        :returns: (array, array, float).
            Vector of predicted mean values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx,xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy,yyd)
        yed = dye if dflag else []
        yef = np.append(ye,yed)
        (x1,x2) = np.meshgrid(xx,xx)
        (x1h1,x2h1) = np.meshgrid(xx,xxd)
        (x1h2,x2h2) = np.meshgrid(xxd,xx)
        (x1d,x2d) = np.meshgrid(xxd,xxd)
        (xs1,xs2) = np.meshgrid(xn,xx)
        (xs1h,xs2h) = np.meshgrid(xn,xxd)
        (xt1,xt2) = np.meshgrid(xn,xn)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1,x2,der=0)
        KKh1 = kk(x1h1,x2h1,der=1)
        KKh2 = kk(x1h2,x2h2,der=-1)
        KKd = kk(x1d,x2d,der=2)
        KK = np.vstack((np.hstack((KKb,KKh2)),np.hstack((KKh1,KKd))))
        LL = spla.cholesky(KK + np.diag(yef**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yf)
        ksb = kk(xs1,xs2,der=-dd) if dd == 1 else kk(xs1,xs2,der=dd)
        ksh = kk(xs1h,xs2h,der=dd+1)
        ks = np.vstack((ksb,ksh))
        vv = np.dot(LL.T,spla.cho_solve((LL,True),ks))
        kt = kk(xt1,xt2,der=2*dd)
        barF = np.dot(ks.T,alpha)          # Mean function
        varF = kt - np.dot(vv.T,vv)        # Variance of mean function

        # Log-marginal-likelihood provides an indication of how statistically well the fit describes the training data
        #    1st term: Describes the goodness of fit for the given data
        #    2nd term: Penalty for complexity / simplicity of the covariance function
        #    3rd term: Penalty for the size of given data set
        lml = -0.5 * np.dot(yf.T,alpha) - lp * np.sum(np.log(np.diag(LL))) - 0.5 * xf.size * np.log(2.0 * np.pi)

        # Log-marginal-likelihood of the null hypothesis (constant at mean value),
        # can be used as a normalization factor for general goodness-of-fit metric
        zfilt = (np.abs(yef) < 1.0e-10)
        yef[zfilt] = 1.0e-10
        lmlz = -0.5 * np.sum(np.power(yf / yef,2.0)) - lp * np.sum(np.log(yef)) - 0.5 * xf.size * np.log(2.0 * np.pi)

        return (barF,varF,lml,lmlz)


    def _gp_brute_deriv1(self,xn,kk,lp,xx,yy,ye):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for brute-force first-order derivative of 1-dimensional Gaussian process regression.
        **Not recommended for production runs**, but useful for testing custom :code:`_Kernel` class implementations
        which have hard-coded derivative calculations.

        :arg xn: array. Vector of x-values at which the fit will be evaluated.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :returns: (array, array, float).
            Vector of predicted mean derivative values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        (x1,x2) = np.meshgrid(xx,xx)
        (xs1,xs2) = np.meshgrid(xx,xn)
        (xt1,xt2) = np.meshgrid(xn,xn)
        # Set up predictive grids with slight offset in x1 and x2, forms corners of a box around original xn point
        step = np.amin(np.abs(np.diff(xn)))
        xnl = xn - step * 0.5e-3            # The step is chosen intelligently to be smaller than smallest dxn
        xnu = xn + step * 0.5e-3
        (xl1,xl2) = np.meshgrid(xx,xnl)
        (xu1,xu2) = np.meshgrid(xx,xnu)
        (xll1,xll2) = np.meshgrid(xnl,xnl)
        (xlu1,xlu2) = np.meshgrid(xnu,xnl)
        (xuu1,xuu2) = np.meshgrid(xnu,xnu)

        KK = kk(x1,x2)
        LL = spla.cholesky(KK + np.diag(ye**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yy)
        # Approximation of first derivative of covf (df/dxn1)
        ksl = kk(xl1,xl2)
        ksu = kk(xu1,xu2)
        dks = (ksu.T - ksl.T) / (step * 1.0e-3)
        dvv = np.dot(LL.T,spla.cho_solve((LL,True),dks))
        # Approximation of second derivative of covf (d^2f/dxn1 dxn2)
        ktll = kk(xll1,xll2)
        ktlu = kk(xlu1,xlu2)
        ktul = ktlu.T
        ktuu = kk(xuu1,xuu2)
        dktl = (ktlu - ktll) / (step * 1.0e-3)
        dktu = (ktuu - ktul) / (step * 1.0e-3)
        ddkt = (dktu - dktl) / (step * 1.0e-3)
        barF = np.dot(dks.T,alpha)          # Mean function
        varF = ddkt - np.dot(dvv.T,dvv)     # Variance of mean function
        lml = -0.5 * np.dot(yy.T,alpha) - lp * np.sum(np.log(np.diag(LL))) - 0.5 * xx.size * np.log(2.0 * np.pi)

        return (barF,varF,lml)


    def _gp_brute_grad_lml(self,kk,lp,xx,yy,ye,dxx,dyy,dye,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for brute-force computation of gradient of log-marginal-likelihood with respect to the
        hyperparameters in logarithmic space. Result must be divided by :code:`ln(10) * theta` in order to have the
        gradient with respect to the hyperparameters in linear space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg dh: float. Step size in hyperparameter space used in derivative approximation.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        """

        xn = np.array([0.0])
        theta = np.log10(kk.hyperparameters)
        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0,theta.size):
            testkk = copy.copy(kk)
            theta_in = theta.copy()
            theta_in[ii] = theta[ii] - 0.5 * dh
            testkk.hyperparameters = np.power(10.0,theta_in)
            llml = itemgetter(2)(self._gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            theta_in[ii] = theta[ii] + 0.5 * dh
            testkk.hyperparameters = np.power(10.0,theta_in)
            ulml = itemgetter(2)(self._gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            gradlml[ii] = (ulml - llml) / dh

        return gradlml


    def _gp_grad_lml(self,kk,lp,xx,yy,ye,dxx,dyy,dye):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for computation of gradient of log-marginal-likelihood with respect to the hyperparameters
        in linear space. Result must be multiplied by :code:`ln(10) * theta` in order to have the gradient with respect
        to the hyperparameters in logarithmic space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        theta = kk.hyperparameters
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx,xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy,yyd)
        yed = dye if dflag else []
        yef = np.append(ye,yed)
        (x1,x2) = np.meshgrid(xx,xx)
        (x1h1,x2h1) = np.meshgrid(xx,xxd)
        (x1h2,x2h2) = np.meshgrid(xxd,xx)
        (x1d,x2d) = np.meshgrid(xxd,xxd)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1,x2,der=0)
        KKh1 = kk(x1h1,x2h1,der=1)
        KKh2 = kk(x1h2,x2h2,der=-1)
        KKd = kk(x1d,x2d,der=2)
        KK = np.vstack((np.hstack((KKb,KKh2)),np.hstack((KKh1,KKd))))
        LL = spla.cholesky(KK + np.diag(yef**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yf)
        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0,theta.size):
            HHb = kk(x1,x2,der=0,hder=ii)
            HHh1 = kk(x1h1,x2h1,der=1,hder=ii)
            HHh2 = kk(x1h2,x2h2,der=-1,hder=ii)
            HHd = kk(x1d,x2d,der=2,hder=ii)
            HH = np.vstack((np.hstack((HHb,HHh2)),np.hstack((HHh1,HHd))))
            PP = np.dot(alpha.T,HH)
            QQ = spla.cho_solve((LL,True),HH)
            gradlml[ii] = 0.5 * np.dot(PP,alpha) - 0.5 * lp * np.sum(np.diag(QQ))

        return gradlml


    def _gp_grad_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Gradient ascent hyperparameter optimization algorithm, searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            theta_step = eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on gradient ascent search.')
        return (newkk,lmlnew)


    def _gp_momentum_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Gradient ascent hyperparameter optimization algorithm with momentum, searches hyperparameters
        in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on momentum gradient ascent search.')
        return (newkk,lmlnew)


    def _gp_nesterov_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Nesterov-accelerated gradient ascent hyperparameter optimization algorithm with momentum,
        searches hyperparameters in log-space. Effectively makes prediction of the next step and
        uses that with back-correction factor as the current update.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        if newkk.is_hderiv_implemented():
            # Hyperparameter derivatives computed in linear space
            gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
            gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_base)
        else:
            gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
        theta_step = eta * gradlml
        theta_old = theta_base.copy()
        theta_new = theta_old + theta_step
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            newkk.hyperparameters = np.power(10.0,theta_new)
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on Nesterov gradient ascent search.')
        return (newkk,lmlnew)


    def _gp_adagrad_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive gradient ascent hyperparameter optimization algorithm, searches hyperparameters
        in log-space. Suffers from extremely aggressive step modification due to continuous
        accumulation of denominator term, recommended to use :code:`adadelta` algorithm.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-2.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        gold = np.zeros(theta_base.shape)
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            gnew = gold + np.power(gradlml,2.0)
            theta_step = eta * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive gradient ascent search.')
        return (newkk,lmlnew)


    def _gp_adadelta_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive gradient ascent hyperparameter optimization algorithm with decaying accumulation
        window, searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Initial guess for gain factor on gradient to define next step, recommended 1.0e-2.

        :arg gam: float. Forgetting factor on accumulated gradient term, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        etatemp = np.ones(theta_base.shape) * eta
        told = theta_step.copy()
        gold = np.zeros(theta_base.shape)
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            gnew = gam * gold + (1.0 - gam) * np.power(gradlml,2.0)
            theta_step = etatemp * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            tnew = gam * told + (1.0 - gam) * np.power(theta_step,2.0)
            etatemp = np.sqrt(tnew + 1.0e-8)
            told = tnew
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on decaying adaptive gradient ascent search.')
        return (newkk,lmlnew)


    def _gp_adam_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive moment estimation hyperparameter optimization algorithm, searches hyperparameters
        in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1))) / (np.sqrt(vnew / (1.0 - b2**(icount + 1))) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive moment estimation search.')
        return (newkk,lmlnew)


    def _gp_adamax_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive moment estimation hyperparameter optimization algorithm with l-infinity, searches
        hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 2.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            unew = b2 * vnew if vold is None else np.nanmax([b2 * vold,np.abs(gradlml)],axis=0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1))) / unew
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive moment l-infinity search.')
        return (newkk,lmlnew)


    def _gp_nadam_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Nesterov-accelerated adaptive moment estimation hyperparameter optimization algorithm,
        searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1)) + (1.0 - b1) * gradlml / (1.0 - b1**(icount + 1))) / (np.sqrt(vnew / (1.0 - b2)) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0,theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on Nesterov adaptive moment search.')
        return (newkk,lmlnew)


    def _condition_data(self,xx,xe,yy,ye,lb,ub,cn):
        """
        **INTERNAL FUNCTION** - Use main call functions!!!

        Conditions the input data to remove data points which are too close together, as
        defined by the user, and data points that are outside user-defined bounds.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg xe: array. Vector of x-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg lb: float. Minimum allowable y-value for input data, values below are omitted from fit procedure.

        :arg ub: float. Maximum allowable y-value for input data, values above are omitted from fit procedure.

        :arg cn: float. Minimum allowable delta-x for input data before applying Gaussian blending.

        :returns: (array, array, array, array, array).
            Vectors in order of conditioned x-values, conditioned x-errors, conditioned y-values, conditioned y-errors,
            number of data points blended into corresponding index.
        """

        good = np.all([np.invert(np.isnan(xx)),np.invert(np.isnan(yy)),np.isfinite(xx),np.isfinite(yy)],axis=0)
        xe = xe[good] if xe.size == xx.size else np.full(xx[good].shape,xe[0])
        ye = ye[good] if ye.size == yy.size else np.full(yy[good].shape,ye[0])
        xx = xx[good]
        yy = yy[good]
        xsc = np.nanmax(np.abs(xx)) if np.nanmax(np.abs(xx)) > 1.0e3 else 1.0   # Scaling avoids overflow when squaring
        ysc = np.nanmax(np.abs(yy)) if np.nanmax(np.abs(yy)) > 1.0e3 else 1.0   # Scaling avoids overflow when squaring
        xx = xx / xsc
        xe = xe / xsc
        yy = yy / ysc
        ye = ye / ysc
        nn = np.array([])
        cxx = np.array([])
        cxe = np.array([])
        cyy = np.array([])
        cye = np.array([])
        for ii in np.arange(0,xx.size):
            if yy[ii] >= lb and yy[ii] <= ub:
                fflag = False
                for jj in np.arange(0,cxx.size):
                    if np.abs(cxx[jj] - xx[ii]) < cn and not fflag:
                        cxe[jj] = np.sqrt((cxe[jj]**2.0 * nn[jj] + xe[ii]**2.0 + cxx[jj]**2.0 * nn[jj] + xx[ii]**2.0) / (nn[jj] + 1.0) - ((cxx[jj] * nn[jj] + xx[ii]) / (nn[jj] + 1.0))**2.0)
                        cxx[jj] = (cxx[jj] * nn[jj] + xx[ii]) / (nn[jj] + 1.0)
                        cye[jj] = np.sqrt((cye[jj]**2.0 * nn[jj] + ye[ii]**2.0 + cyy[jj]**2.0 * nn[jj] + yy[ii]**2.0) / (nn[jj] + 1.0) - ((cyy[jj] * nn[jj] + yy[ii]) / (nn[jj] + 1.0))**2.0)
                        cyy[jj] = (cyy[jj] * nn[jj] + yy[ii]) / (nn[jj] + 1.0)
                        nn[jj] = nn[jj] + 1.0
                        fflag = True
                if not fflag:
                    nn = np.hstack((nn,1.0))
                    cxx = np.hstack((cxx,xx[ii]))
                    cxe = np.hstack((cxe,xe[ii]))
                    cyy = np.hstack((cyy,yy[ii]))
                    cye = np.hstack((cye,ye[ii]))
        cxx = cxx * xsc
        cxe = cxe * xsc
        cyy = cyy * ysc
        cye = cye * ysc
        return (cxx,cxe,cyy,cye,nn)


    def __basic_fit(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,dxdata=None,dydata=None,dyerr=None,epsilon=None,method=None,spars=None,sdiff=None,do_drv=False,rtn_cov=False):
        """
        **RESTRICTED ACCESS FUNCTION** - Can be called externally for testing if user is familiar with algorithm.

        Basic GP regression fitting routine, **recommended** to call this instead of the bare-bones functions
        as this applies additional input checking.

        .. note::

            This function does **not** strictly use class data and does **not** store results inside the class
            either!!! This is done to allow this function to be used as a minimal working version and also as
            a standalone test for new :code:`_Kernel`, :code:`_OperatorKernel` and :code:`_WarpingFunction`
            class implementations.

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. Vector of x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. Vector of dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. Vector of dy/dx-errors of derivative data points to be included in fit, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable. (optional)

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include::
                       [:code:`grad`, :code:`mom`, :code:`nag`, :code:`adagrad`, :code:`adadelta`, :code:`adam`, :code:`adamax`, :code:`nadam`] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2. (optional)

        :kwarg do_drv: bool. Set as true to predict the derivative of the fit instead of the fit. (optional)

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors. (optional)

        :returns: (array, array, float, object).
            Vector of predicted mean values, vector or matrix of predicted errors, log-marginal-likelihood of fit
            including the regularization component, final :code:`_Kernel` instance with optimized hyperparameters if performed.
        """

        xn = None
        kk = copy.copy(self._kk)
        lp = self._lp
        xx = copy.deepcopy(self._xx)
        yy = copy.deepcopy(self._yy)
        ye = copy.deepcopy(self._ye) if self._gpye is None else copy.deepcopy(self._gpye)
        dxx = copy.deepcopy(self._dxx)
        dyy = copy.deepcopy(self._dyy)
        dye = copy.deepcopy(self._dye)
        eps = self._eps
        opm = self._opm
        opp = copy.deepcopy(self._opp)
        dh = self._dh
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        midx = None
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,_Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            lp = float(regpar)
        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        elif isinstance(yerr,str):
            ye = None
        if isinstance(dxdata,(list,tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dxx = temp.flatten()
        elif isinstance(dxdata,np.ndarray) and dxdata.size > 0:
            dxx = dxdata.flatten()
        elif isinstance(dxdata,str):
            dxx = None
        if isinstance(dydata,(list,tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dyy = temp.flatten()
        elif isinstance(dydata,np.ndarray) and dydata.size > 0:
            dyy = dydata.flatten()
        elif isinstance(dydata,str):
            dyy = None
        if isinstance(dyerr,(list,tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dye = temp.flatten()
        elif isinstance(dyerr,np.ndarray) and dyerr.size > 0:
            dye = dyerr.flatten()
        elif isinstance(dyerr,str):
            dye = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            eps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            eps = None
        elif isinstance(epsilon,str):
            eps = None
        if isinstance(method,str):
            mstr = method.lower()
            if mstr in self._opopts:
                midx = self._opopts.index(mstr)
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) >= 0 and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                opm = self._opopts[1]
                oppt = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 2:
                opm = self._opopts[2]
                oppt = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 3:
                opm = self._opopts[3]
                oppt = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 4:
                opm = self._opopts[4]
                oppt = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 5:
                opm = self._opopts[5]
                oppt = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 6:
                opm = self._opopts[6]
                oppt = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 7:
                opm = self._opopts[7]
                oppt = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            else:
                opm = self._opopts[0]
                oppt = np.array([1.0e-5]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    opp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    opp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            dh = float(sdiff)

        barF = None
        errF = None
        lml = None
        lmlz = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,_Kernel):
            # Remove all data and associated data that contain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx,xe,yy,ye,nn) = self._condition_data(xx,xe,yy,ye,lb,ub,cn)
            myy = np.mean(yy)
            yy = yy - myy
            sc = np.nanmax(np.abs(yy))
            if sc == 0.0:
                sc = 1.0
            yy = yy / sc
            ye = ye / sc
            dnn = None
            if dxx is not None and dyy is not None and dxx.size == dyy.size:
                if dye is None:
                    dye = np.array([0.0])
                dxe = np.array([0.0])
                (dxx,dxe,dyy,dye,dnn) = self._condition_data(dxx,dxe,dyy,dye,-1.0e50,1.0e50,cn)
                dyy = dyy / sc
                dye = dye / sc
            dd = 1 if do_drv else 0
            nkk = copy.copy(kk)
            if eps is not None and not do_drv:
                if opm == 'mom' and opp.size > 1:
                    (nkk,lml) = self._gp_momentum_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'nag' and opp.size > 1:
                    (nkk,lml) = self._gp_nesterov_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'adagrad' and opp.size > 0:
                    (nkk,lml) = self._gp_adagrad_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],dh)
                elif opm == 'adadelta' and opp.size > 1:
                    (nkk,lml) = self._gp_adadelta_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'adam' and opp.size > 2:
                    (nkk,lml) = self._gp_adam_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'adamax' and opp.size > 2:
                    (nkk,lml) = self._gp_adamax_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'nadam' and opp.size > 2:
                    (nkk,lml) = self._gp_nadam_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'grad' and opp.size > 0:
                    (nkk,lml) = self._gp_grad_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],dh)
            (barF,varF,lml,lmlz) = self._gp_base_alg(xn,nkk,lp,xx,yy,ye,dxx,dyy,dye,dd)
            barF = barF * sc if do_drv else barF * sc + myy
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF,errF,lml,lmlz,nkk)


    def __brute_derivative(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,rtn_cov=False):
        """
        **RESTRICTED ACCESS FUNCTION** - Can be called externally for testing if user is familiar with algorithm.

        Brute-force numerical GP regression derivative routine, **recommended** to call this instead of bare-bones
        functions above. Kept for ability to convince user of validity of regular GP derivative, but can also be
        wildly wrong on some data due to numerical errors.

        .. note::

            *Recommended* to use derivative flag on :code:`__basic_fit()` function, as it was tested to be
            more robust, provided the input kernels are properly defined.

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors. (optional)

        :returns: (array, array, float).
            Vector of predicted dy/dx-values, vector or matrix of predicted dy/dx-errors, log-marginal-likelihood of fit
            including the regularization component.
        """

        xn = None
        kk = self._kk
        lp = self._lp
        xx = self._xx
        yy = self._yy
        ye = self._ye if self._gpye is None else self._gpye
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,_Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar,(float,int)) and float(regpar) > 0.0:
            self._lp = float(regpar)
        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        if ye is None and yy is not None:
            ye = np.zeros(yy.shape)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,_Kernel):
            # Remove all data and associated data that conatain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx,xe,yy,ye,nn) = self._condition_data(xx,xe,yy,ye,lb,ub,cn)
            myy = np.mean(yy)
            yy = yy - myy
            sc = np.nanmax(np.abs(yy))
            if sc == 0.0:
                sc = 1.0
            yy = yy / sc
            ye = ye / sc
            (barF,varF,lml) = self._gp_brute_deriv1(xn,kk,lp,xx,yy,ye)
            barF = barF * sc
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF,errF,lml)


    def make_HSGP_errors(self):
        """
        Calculates a vector of modified y-errors based on GPR fit of input y-errors,
        for use inside a heteroscedastic GPR execution.

        .. note::

            This function is automatically called inside :code:`GPRFit()` when the
            :code:`hsgp_flag` argument is :code:`True`. For this reason, this function
            automatically stores all results within the appropriate class variables.

        :returns: none.
        """

        if isinstance(self._ekk,_Kernel) and self._ye is not None and self._yy.size == self._ye.size:
            elml = None
            ekk = None
            xntest = np.array([0.0])
            ye = copy.deepcopy(self._ye) if self._gpye is None else copy.deepcopy(self._gpye)
            aye = np.full(ye.shape,np.nanmax([0.2 * np.mean(np.abs(ye)),1.0e-3 * np.nanmax(np.abs(self._yy))]))
#            dye = copy.deepcopy(self._dye)
#            adye = np.full(dye.shape,np.nanmax([0.2 * np.mean(np.abs(dye)),1.0e-3 * np.nanmax(np.abs(self._dyy))])) if dye is not None else None
#            if adye is not None:
#                adye[adye < 1.0e-2] = 1.0e-2
            if self._ekk.bounds is not None and self._eeps is not None and self._egpye is None:
                elp = self._elp
                ekk = copy.copy(self._ekk)
                ekkvec = []
                elmlvec = []
                try:
                    (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
#                    (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dydata=dye,dyerr=adye,epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                    ekkvec.append(copy.copy(ekk))
                    elmlvec.append(elml)
                except (ValueError,np.linalg.linalg.LinAlgError):
                    ekkvec.append(None)
                    elmlvec.append(np.NaN)
                for jj in np.arange(0,self._enr):
                    ekb = np.log10(self._ekk.bounds)
                    etheta = np.abs(ekb[1,:] - ekb[0,:]).flatten() * np.random.random_sample((ekb.shape[1],)) + np.nanmin(ekb,axis=0).flatten()
                    ekk.hyperparameters = np.power(10.0,etheta)
                    try:
                        (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
#                        (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dydata=dye,dyerr=adye,epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                        ekkvec.append(copy.copy(ekk))
                        elmlvec.append(elml)
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ekkvec.append(None)
                        elmlvec.append(np.NaN)
                eimaxv = np.where(elmlvec == np.nanmax(elmlvec))[0]
                if len(eimaxv) > 0:
                    eimax = eimaxv[0]
                    ekk = ekkvec[eimax]
                    self._ekk = copy.copy(ekkvec[eimax])
                else:
                    raise ValueError('None of the error fit attempts converged. Please change error kernel settings and try again.')
            elif self._eeps is not None and self._egpye is None:
                elp = self._elp
                ekk = copy.copy(self._ekk)
                (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
#                (elml,ekk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=aye,dydata=dye,dyerr=adye,epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                self._ekk = copy.copy(ekk)
            if isinstance(self._ekk,_Kernel):
                epsx = 1.0e-8 * (np.nanmax(self._xx) - np.nanmin(self._xx)) if self._xx.size > 1 else 1.0e-8
                xntest = self._xx.copy() + epsx
                tgpye = itemgetter(0)(self.__basic_fit(xntest,kernel=self._ekk,regpar=self._elp,ydata=ye,yerr=aye,dxdata='None',dydata='None',dyerr='None',epsilon='None'))
#                self._gpye = itemgetter(0)(self.__basic_fit(xntest,kernel=self._ekk,regpar=self._elp,ydata=ye,yerr=aye,dydata=dye,dyerr=adye,epsilon='None'))
                self._gpye = np.abs(tgpye)
                self._egpye = aye.copy()
        else:
            raise ValueError('Check input y-errors to make sure they are valid.')


    def make_NIGP_errors(self,nrestarts=0,hsgp_flag=False):
        """
        Calculates a vector of modified y-errors based on input x-errors and a test model
        gradient, for use inside a noisy input GPR execution.

        .. note::

            This function is automatically called inside :code:`GPRFit()` when the
            :code:`nigp_flag` argument is :code:`True`. For this reason, this function
            automatically stores all results within the appropriate class variables.

        .. warning::

            This function does not iterate until the test model derivatives and the actual
            fit derivatives are self-consistent! Although this would be the most rigourous
            implementation, it was decided that this approximation was good enough for the
            uses of the current implementation. (v >= 1.0.1)

        .. warning::

            The results of this function may be washed away by the heteroscedastic GP
            implementation due to the fact that the y-error modifications are included
            when fitting the error kernel. This can be addressed in the future by
            separating the contributions due to noisy input and due to heteroscedastic
            GP, with a separate noise kernel for each.

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within the provided hyperparameter bounds. (optional)

        :kwarg hsgp_flag: bool. Indicates Gaussian Process regression fit with variable y-errors. (optional)

        :returns: none.
        """

        # Check inputs
        nr = 0
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nrestarts) > 0:
            nr = int(nrestarts)

        if isinstance(self._kk,_Kernel) and self._xe is not None and self._xx.size == self._xe.size:
            nlml = None
            nkk = None
            xntest = np.array([0.0])
            if not isinstance(self._nikk,_Kernel):
                if self._kk.bounds is not None and nr > 0:
                    tkk = copy.copy(self._kk)
                    kkvec = []
                    lmlvec = []
                    try:
                        (tlml,tkk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=tkk))
                        kkvec.append(copy.copy(tkk))
                        lmlvec.append(tlml)
                    except ValueError:
                        kkvec.append(None)
                        lmlvec.append(np.NaN)
                    for ii in np.arange(0,nr):
#                        kb = self._kb
                        kb = np.log10(self._kk.bounds)
                        theta = np.abs(kb[1,:] - kb[0,:]).flatten() * np.random.random_sample((kb.shape[1],)) + np.nanmin(kb,axis=0).flatten()
                        tkk.hyperparameters = np.power(10.0,theta)
                        try:
                            (tlml,tkk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=tkk))
                            kkvec.append(copy.copy(tkk))
                            lmlvec.append(tlml)
                        except ValueError:
                            kkvec.append(None)
                            lmlvec.append(np.NaN)
                    imax = np.where(lmlvec == np.nanmax(lmlvec))[0][0]
                    (nlml,nkk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=kkvec[imax],epsilon='None'))
                else:
                    (nlml,nkk) = itemgetter(2,4)(self.__basic_fit(xntest))
            else:
                nkk = copy.copy(self._nikk)
            if isinstance(nkk,_Kernel):
                self._nikk = copy.copy(nkk)
                epsx = 1.0e-8 * (np.nanmax(self._xx) - np.nanmin(self._xx)) if self._xx.size > 1 else 1.0e-8
                xntest = self._xx.copy() + epsx
                dbarF = itemgetter(0)(self.__basic_fit(xntest,kernel=nkk,do_drv=True))
#                cxe = self._xe.copy()
#                cxe[np.isnan(cxe)] = 0.0
#                self._gpxe = np.abs(cxe * dbarF)
                cxe = self._xe.copy()
                cye = self._ye.copy() if self._gpye is None else self._gpye.copy()
                nfilt = np.any([np.isnan(cxe),np.isnan(cye)],axis=0)
                cxe[nfilt] = 0.0
                cye[nfilt] = 0.0
                self._gpye = np.sqrt(cye**2.0 + (cxe * dbarF)**2.0)
                self._egpye = np.full(cye.shape,np.nanmax([0.2 * np.mean(np.abs(self._gpye)),1.0e-3 * np.nanmax(np.abs(self._yy))])) if not hsgp_flag else self._egpye
        else:
            raise ValueError('Check input x-errors to make sure they are valid.')


    def GPRFit(self,xnew,hsgp_flag=True,nigp_flag=False,nrestarts=None):
        """
        Main GP regression fitting routine, **recommended** to call this after using set functions, instead of the
        :code:`__basic_fit()` function, as this adapts the method based on inputs, performs 1st derivative and
        saves output to class variables.

        - Includes implementation of Monte Carlo kernel restarts within the user-defined bounds, via nrestarts argument
        - Includes implementation of Heteroscedastic Output Noise, requires setting of error kernel before fitting
            For details, see article: K. Kersting, 'Most Likely Heteroscedastic Gaussian Process Regression' (2007)
        - Includes implementation of Noisy-Input Gaussian Process (NIGP) assuming Gaussian x-error, via nigp_flag argument
            For details, see article: A. McHutchon, C.E. Rasmussen, 'Gaussian Process Training with Input Noise' (2011)

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg hsgp_flag: bool. Set as true to perform Gaussian Process regression fit with proper propagation of y-errors. Default is :code:`True`. (optional)

        :kwarg nigp_flag: bool. Set as true to perform Gaussian Process regression fit with proper propagation of x-errors. Default is :code:`False`. (optional)

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within the provided hyperparameter bounds. (optional)

        :returns: none.
        """
        # Check inputs
        xn = None
        nr = 0
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nrestarts) > 0:
            nr = int(nrestarts)
        if xn is None:
            raise ValueError('A valid vector of prediction x-points must be given.')
        oxn = copy.deepcopy(xn)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        barF = None
        varF = None
        lml = None
        lmlz = None
        nkk = None
        estF = None
        if nigp_flag:
            self.make_NIGP_errors(nr,hsgp_flag=hsgp_flag)
        if hsgp_flag:
            self.make_HSGP_errors()
        if self._gpye is None:
            hsgp_flag = False
            nigp_flag = False
            self._gpye = self._ye.copy()
            self._egpye = None

        # These loops adjust overlapping values between raw data vector and requested prediction vector, to avoid NaN values in final prediction
        if self._xx is not None:
            epsx = 1.0e-6 * (np.nanmax(xn) - np.nanmin(xn)) if xn.size > 1 else 1.0e-6 * (np.nanmax(self._xx) - np.nanmin(self._xx))
            for xi in np.arange(0,xn.size):
                for rxi in np.arange(0,self._xx.size):
                    if xn[xi] == self._xx[rxi]:
                        xn[xi] = xn[xi] + epsx
        if self._dxx is not None:
            epsx = 1.0e-6 * (np.nanmax(xn) - np.nanmin(xn)) if xn.size > 1 else 1.0e-6 * (np.nanmax(self._dxx) - np.nanmin(self._dxx))
            for xi in np.arange(0,xn.size):
                for rxi in np.arange(0,self._dxx.size):
                    if xn[xi] == self._dxx[rxi]:
                        xn[xi] = xn[xi] + epsx

        if self._egpye is not None:
            edye = np.full(self._dye.shape,np.nanmax([0.2 * np.mean(np.abs(self._dye)),1.0e-3 * np.nanmax(np.abs(self._dyy))])) if self._dye is not None else None
            if edye is not None:
                edye[edye < 1.0e-2] = 1.0e-2
            (self._barE,self._varE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dxdata='None',dydata='None',dyerr='None',epsilon='None',rtn_cov=True))
            (self._dbarE,self._dvarE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dxdata='None',dydata='None',dyerr='None',do_drv=True,rtn_cov=True))
#            (self._barE,self._varE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dydata=self._dye,dyerr=edye,epsilon='None',rtn_cov=True))
#            (self._dbarE,self._dvarE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dydata=self._dye,dyerr=edye,do_drv=True,rtn_cov=True))

#            nxn = np.linspace(np.nanmin(xn),np.nanmax(xn),1000)
#            ddx = np.nanmin(np.diff(nxn)) * 1.0e-2
#            xnl = nxn - 0.5 * ddx
#            xnu = nxn + 0.5 * ddx
#            dbarEl = itemgetter(0)(self.__basic_fit(xnl,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dxdata='None',dydata='None',dyerr='None',do_drv=True))
#            dbarEu = itemgetter(0)(self.__basic_fit(xnu,kernel=self._ekk,ydata=self._gpye,yerr=self._egpye,dxdata='None',dydata='None',dyerr='None',do_drv=True))
#            ddbarEt = np.abs(dbarEu - dbarEl) / ddx
#            nsum = 50
#            ddbarE = np.zeros(xn.shape)
#            for nx in np.arange(0,xn.size):
#                ivec = np.where(nxn >= xn[nx])[0][0]
#                nbeg = nsum - (ivec + 1) if (ivec + 1) < nsum else 0
#                nend = nsum - (nxn.size - ivec - 1) if (nxn.size - ivec - 1) < nsum else 0
#                temp = None
#                if nbeg > 0:
#                    vbeg = np.full((nbeg,),ddbarEt[0])
#                    temp = np.hstack((vbeg,ddbarEt[:ivec+nsum+1]))
#                    ddbarE[nx] = float(np.mean(temp))
#                elif nend > 0:
#                    vend = np.full((nend,),ddbarEt[-1]) if nend > 0 else np.array([])
#                    temp = np.hstack((ddbarEt[ivec-nsum:],vend))
#                    ddbarE[nx] = float(np.mean(temp))
#                else:
#                    ddbarE[nx] = float(np.mean(ddbarEt[ivec-nsum:ivec+nsum+1]))
#            self._ddbarE = ddbarE.copy()
        else:
            self._gpye = np.full(xn.shape,np.sqrt(np.nanmean(np.power(self._ye,2.0)))) if self._ye is not None else None
            self._barE = copy.deepcopy(self._gpye) if self._gpye is not None else None
            self._varE = np.zeros(xn.shape) if self._barE is not None else None
            self._dbarE = np.zeros(xn.shape) if self._barE is not None else None
            self._dvarE = np.zeros(xn.shape) if self._barE is not None else None
#            self._ddbarE = np.zeros(xn.shape) if self._barE is not None else None

        if isinstance(self._kk,_Kernel) and self._kk.bounds is not None and nr > 0:
            xntest = np.array([0.0])
            tkk = copy.copy(self._kk)
            kkvec = []
            lmlvec = []
            try:
                (tlml,tkk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=tkk))
                kkvec.append(copy.copy(tkk))
                lmlvec.append(tlml)
            except (ValueError,np.linalg.linalg.LinAlgError):
                kkvec.append(None)
                lmlvec.append(np.NaN)
            for ii in np.arange(0,nr):
#                kb = self._kb
                kb = np.log10(self._kk.bounds)
                theta = np.abs(kb[1,:] - kb[0,:]).flatten() * np.random.random_sample((kb.shape[1],)) + np.nanmin(kb,axis=0).flatten()
                tkk.hyperparameters = np.power(10.0,theta)
                try:
                    (tlml,tkk) = itemgetter(2,4)(self.__basic_fit(xntest,kernel=tkk))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except (ValueError,np.linalg.linalg.LinAlgError):
                    kkvec.append(None)
                    lmlvec.append(np.NaN)
            imaxv = np.where(lmlvec == np.nanmax(lmlvec))[0]
            if len(imaxv) > 0:
                imax = imaxv[0]
                (barF,varF,lml,lmlz,nkk) = self.__basic_fit(xn,kernel=kkvec[imax],epsilon='None',rtn_cov=True)
                estF = itemgetter(0)(self.__basic_fit(self._xx + 1.0e-10,kernel=kkvec[imax],epsilon='None',rtn_cov=True))
            else:
                raise ValueError('None of the fit attempts converged. Please adjust kernel settings and try again.')
        elif isinstance(self._kk,_Kernel):
            (barF,varF,lml,lmlz,nkk) = self.__basic_fit(xn,rtn_cov=True)
            estF = itemgetter(0)(self.__basic_fit(self._xx + 1.0e-10,kernel=nkk,epsilon='None',rtn_cov=True))

        if barF is not None and isinstance(nkk,_Kernel):
            self._xF = copy.deepcopy(oxn)
            self._barF = copy.deepcopy(barF)
            self._varF = copy.deepcopy(varF) if varF is not None else None
            self._estF = copy.deepcopy(estF) if estF is not None else None
            self._lml = lml
            self._nulllml = lmlz
            self._kk = copy.copy(nkk) if isinstance(nkk,_Kernel) else None
            (dbarF,dvarF) = itemgetter(0,1)(self.__basic_fit(xn,do_drv=True,rtn_cov=True))
            self._dbarF = copy.deepcopy(dbarF) if dbarF is not None else None
            self._dvarF = copy.deepcopy(dvarF) if dvarF is not None else None
            self._varN = np.diag(np.power(self._barE,2.0)) if self._barE is not None else np.diag(np.zeros(self._xF.shape))
            self._dvarN = np.diag(np.power(self._dbarE,2.0)) if self._dbarE is not None else np.diag(np.zeros(self._xF.shape))

            # It seems that the second derivative term is not necessary, should be used to refine the mathematics!
#            ddfac = copy.deepcopy(self._ddbarE) if self._ddbarE is not None else 0.0
#            self._dvarN = np.diag(2.0 * (np.power(self._dbarE,2.0) + np.abs(self._barE * ddfac))) if self._dbarE is not None else None
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)


    def sample_GP(self,nsamples,actual_noise=False,without_noise=False,simple_out=False):
        """
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        :code:`GPRFit()` method.

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance as actual Gaussian noise. Only operates on diagonal elements. (optional)

        :kwarg without_noise: bool. Specifies complete exclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation. (optional)

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest :code:`GPRFit()`. If :code:`simple_out = True`, row 0 is the mean and row 1 is the 1 sigma error.
        """

        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        samples = None
        if ns > 0:
            noise_flag = actual_noise if not without_noise else False
            mu = self.get_gp_mean()
            var = self.get_gp_variance(noise_flag=noise_flag)
            mult_flag = not actual_noise if not without_noise else False
            mult = self.get_gp_std(noise_flag=mult_flag) / self.get_gp_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu,cov=var,size=ns)
            samples = mult * (samples - mu) + mu
            if samples is not None and simple_out:
                mean = np.nanmean(samples,axis=0)
                std = np.nanstd(samples,axis=0)
                samples = np.vstack((mean,std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return samples


    def sample_GP_derivative(self,nsamples,actual_noise=False,without_noise=False,simple_out=False):
        """
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        :code:`GPRFit()` method.

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance as actual Gaussian noise. Only operates on diagonal elements. (optional)

        :kwarg without_noise: bool. Specifies complete exclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation. (optional)

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest :code:`GPRFit()`. If :code:`simple_out = True`, row 0 is the mean and row 1 is the 1 sigma error.
        """

        # Check instantiation of output class variables
        if self._xF is None or self._dbarF is None or self._dvarF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        samples = None
        if ns > 0:
            noise_flag = actual_noise if not without_noise else False
            mu = self.get_gp_drv_mean()
            var = self.get_gp_drv_variance(noise_flag=noise_flag)
            mult_flag = not actual_noise if not without_noise else False
            mult = self.get_gp_drv_std(noise_flag=mult_flag) / self.get_gp_drv_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu,cov=var,size=ns)
            samples = mult * (samples - mu) + mu
            if samples is not None and simple_out:
                mean = np.nanmean(samples,axis=0)
                std = np.nanstd(samples,axis=0)
                samples = np.vstack((mean,std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return samples


    def MCMC_posterior_sampling(self,nsamples):
        """
        Performs Monte Carlo Markov chain based posterior analysis over hyperparameters,
        using the log-marginal-likelihood as the acceptance criterion.

        .. warning::

            This function is suspected to be **incorrect** as currently coded! It should
            use data likelihood from model as the acceptance criterion instead of the
            log-marginal-likelihood. However, MCMC analysis was found only to be
            necessary when using non-Gaussian likelihoods and priors, otherwise the
            result is effectively equivalent to maximization of the LML. (v >= 1.0.1)

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :returns: (array, array, array, array).
            Matrices containing rows of predicted y-values, predicted y-errors, predicted dy/dx-values,
            predicted dy/dx-errors with each having a number of rows equal to the number of samples.
        """
        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to use MCMC posterior sampling.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        sbarM = None
        ssigM = None
        sdbarM = None
        sdsigM = None
        if isinstance(self._kk,_Kernel) and ns > 0:
            olml = self._lml
            otheta = np.log10(self._kk.hyperparameters)
            tlml = olml
            theta = otheta.copy()
            step = np.ones(theta.shape)
            flagvec = [True] * theta.size
            for ihyp in np.arange(0,theta.size):
                xntest = np.array([0.0])
                iflag = flagvec[ihyp]
                while iflag:
                    tkk = copy.copy(self._kk)
                    theta_step = np.zeros(theta.shape)
                    theta_step[ihyp] = step[ihyp]
                    theta_new = theta + theta_step
                    tkk.hyperparameters = np.power(10.0,theta_new)
                    ulml = None
                    try:
                        ulml = itemgetter(2)(self.__basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ulml = tlml - 3.0
                    theta_new = theta - theta_step
                    tkk.hyperparameters = np.power(10.0,theta_new)
                    llml = None
                    try:
                        llml = itemgetter(2)(self.__basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        llml = tlml - 3.0
                    if (ulml - tlml) >= -2.0 or (llml - tlml) >= -2.0:
                        iflag = False
                    else:
                        step[ihyp] = 0.5 * step[ihyp]
                flagvec[ihyp] = iflag
            nkk = copy.copy(self._kk)
            for ii in np.arange(0,ns):
                theta_prop = theta.copy()
                accept = False
                xntest = np.array([0.0])
                nlml = tlml
                jj = 0
                kk = 0
                while not accept:
                    jj = jj + 1
                    rstep = np.random.normal(0.0,0.5*step)
                    theta_prop = theta_prop + rstep
                    nkk.hyperparameters = np.power(10.0,theta_prop)
                    try:
                        nlml = itemgetter(2)(self.__basic_fit(xntest,kernel=nkk,epsilon='None'))
                        if (nlml - tlml) > 0.0:
                            accept = True
                        else:
                            accept = True if np.power(10.0,nlml - tlml) >= np.random.uniform() else False
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        accept = False
                    if jj > 100:
                        step = 0.9 * step
                        jj = 0
                        kk = kk + 1
                    if kk > 100:
                        theta_prop = otheta.copy()
                        tlml = olml
                        kk = 0
                tlml = nlml
                theta = theta_prop.copy()
                xn = self._xF.copy()
                nkk.hyperparameters = np.power(10.0,theta)
                (barF,sigF,tlml,tlmlz,nkk) = self.__basic_fit(xn,kernel=nkk,epsilon='None')
                sbarM = barF.copy() if sbarM is None else np.vstack((sbarM,barF))
                ssigM = sigF.copy() if ssigM is None else np.vstack((ssigM,sigF))
                (dbarF,dsigF) = itemgetter(0,1)(self.__basic_fit(xn,kernel=nkk,epsilon='None',do_drv=True))
                sdbarM = dbarF.copy() if sdbarM is None else np.vstack((sdbarM,dbarF))
                sdsigM = dsigF.copy() if sdsigM is None else np.vstack((sdsigM,dsigF))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return (sbarM,ssigM,sdbarM,sdsigM)



# ****************************************************************************************************************************************
# ------- Some helpful functions and classes for reproducability and user-friendliness ---------------------------------------------------
# ****************************************************************************************************************************************


class SimplifiedGaussianProcessRegression1D(GaussianProcessRegression1D):
    """
    A simplified version of the main :code:`GaussianProcessRegression1D`
    class with pre-defined settings, only requiring the bare necessities
    to use for fitting. Although this class allows an entry point akin
    to typical :mod:`scipy` classes, it is provided primarily to be
    used as a template for implementations meant to simplify the GPR1D
    experience for the average user.

    .. note::

        Optimization of hyperparameters is only performed *once* using
        settings at the time of the *first* call! All subsequent calls
        use the results of the first optimization, regardless of its
        quality or convergence status.

    :arg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in the fit procedure.

    :arg xdata: array. Vector of x-values corresponding to data to be fitted.

    :arg ydata: array. Vector of y-values corresponding to data to be fitted. Must be the same shape as :code:`xdata`.

    :arg yerr: array. Vector of y-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`.

    :kwarg xerr: array. Vector of x-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`. (optional)

    :kwarg kernel_bounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

    :kwarg reg_par: float. Parameter adjusting penalty on kernel complexity. (optional)

    :kwarg epsilon: float. Convergence criterion on change in log-marginal-likelihood. (optional)

    :kwarg num_restarts: int. Number of kernel restarts. (optional)

    :kwarg hyp_opt_gain: float. Gain value on the hyperparameter optimizer, expert use only. (optional)
    """

    def __init__(self,kernel,xdata,ydata,yerr,xerr=None,kernel_bounds=None,reg_par=1.0,epsilon=1.0e-2,num_restarts=0,hyp_opt_gain=1.0e-2,include_noise=True):
        """
        Defines customized :code:`GaussianProcessRegression1D` instance with
        a pre-defined common settings for both data fit and error fit. Input
        parameters reduced only to essentials and most crucial knobs for
        fine-tuning.

        :arg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in the fit procedure.

        :arg xdata: array. Vector of x-values corresponding to data to be fitted.

        :arg ydata: array. Vector of y-values corresponding to data to be fitted. Must be the same shape as :code:`xdata`.

        :arg yerr: array. Vector of y-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`.

        :kwarg xerr: array. Optional vector of x-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`. (optional)

        :kwarg kernel_bounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :arg reg_par: float. Parameter adjusting penalty on kernel complexity. (optional)

        :kwarg epsilon: float. Convergence criterion on change in log-marginal-likelihood. (optional)

        :kwarg num_restarts: int. Number of kernel restarts. (optional)

        :kwarg hyp_opt_gain: float. Gain value on the hyperparameter optimizer, expert use only. (optional)

        :kwarg include_noise: bool. Specifies inclusion of Gaussian noise term in returned variance. Only operates on diagonal elements. (optional)

        :returns: none.
        """
        super(SimplifiedGaussianProcessRegression1D,self).__init__()
        self._nrestarts = num_restarts

        self.set_raw_data(xdata=xdata,ydata=ydata,yerr=yerr,xerr=xerr)

        eps = 'none' if not isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) else epsilon
        sg = hyp_opt_gain if isinstance(hyp_opt_gain,(float,int,np_itypes,np_utypes,np_ftypes)) else 1.0e-1
        self.set_kernel(kernel=kernel,kbounds=kernel_bounds,regpar=reg_par)
        self.set_search_parameters(epsilon=epsilon,method='adam',spars=[sg,0.4,0.8])

        self._perform_heterogp = False if isinstance(yerr,(float,int,np_itypes,np_utypes,np_ftypes)) or self._ye is None else True
        self._perform_nigp = False if self._xe is None else True
        self._include_noise = True if include_noise else False

        if self._perform_heterogp:
            error_length = 5.0 * (np.nanmax(self._xx) - np.nanmin(self._xx)) / float(self._xx.size) if self._xx is not None else 5.0e-1
            error_kernel = RQ_Kernel(5.0e-1,error_length,3.0e1)
            error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1,error_length * 0.25,1.0e1],[1.0e0,error_length * 4.0,5.0e1]])
            self.set_error_kernel(kernel=error_kernel,kbounds=error_kernel_hyppar_bounds,regpar=5.0,nrestarts=0)
            self.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])
        elif isinstance(yerr,(float,int,np_itypes,np_utypes,np_ftypes)):
            ye = np.full(self._yy.shape,yerr)
            self.set_raw_data(yerr=ye)


    def __call__(self,xnew):
        """
        Defines a simplified fitting execution, *only* performs
        optimization on the *first* call. Subsequent calls
        merely evaluate the optimized fit at the input x-values.

        :arg xnew: array. Vector of x-values corresponding to points at which the GPR results should be evaulated.

        :returns: tuple.
                  Mean of GPR predictive distribution, ie. the fit ;
                  Standard deviation of mean, given as 1 sigma ;
                  Mean derivative of GPR predictive disstribution, ie. the derivative of the fit ;
                  Standard deviation of mean derivative, given as 1 sigma.
        """
        nrestarts = self._nrestarts
        if self._xF is not None:
            nrestarts = None
            self.set_search_parameters(epsilon='none')
            self.set_error_search_parameters(epsilon='none')
        self.GPRFit(xnew,hsgp_flag=self._perform_heterogp,nigp_flag=self._perform_nigp,nrestarts=nrestarts)
        return self.get_gp_results(noise_flag=self._include_noise)


    def sample(self,xnew,derivative=False):
        """
        Provides a more intuitive function for sampling the
        predictive distribution. Only provides one sample
        per call, unlike the more complex function in the
        main class.

        :arg xnew: array. Vector of x-values corresponding to points where GPR results should be evaulated at.

        :kwarg derivative: bool. Flag to indicate sampling of fit derivative instead of the fit. (optional)

        :returns: array. Vector of y-values corresponding to a random sample of the GPR predictive distribution.
        """
        self.__call__(xnew)
        remove_noise = not self._include_noise
        output = self.sample_GP(1,actual_noise=False,without_noise=remove_noise) if not derivative else self.sample_GP_derivative(1,actual_noise=False,without_noise=remove_noise)
        return output


def KernelConstructor(name):
    """
    Function to construct a kernel solely based on the kernel codename.

    .. note::

        All :code:`_OperatorKernel` class implementations should use encapsulating
        round brackets to specify their constituents. (v >= 1.0.1)

    :arg name: str. The codename of the desired :code:`_Kernel` instance.

    :returns: object. The desired :code:`_Kernel` instance with default parameters. Returns :code:`None` if given kernel codename was invalid.
    """

    kernel = None
    if isinstance(name,str):
        m = re.search(r'^(.*?)\((.*)\)$',name)
        if m:
            links = m.group(2).split('-')
            names = []
            bflag = False
            rname = ''
            for jj in np.arange(0,len(links)):
                rname = links[jj] if not bflag else rname + '-' + links[jj]
                if re.search('\(',links[jj]):
                    bflag = True
                if re.search('\)',links[jj]):
                    bflag = False
                if not bflag:
                    names.append(rname)
            kklist = []
            for ii in np.arange(0,len(names)):
                kklist.append(KernelConstructor(names[ii]))
            if re.search('^Sum$',m.group(1)):
                kernel = Sum_Kernel(klist=kklist)
            elif re.search('^Prod$',m.group(1)):
                kernel = Product_Kernel(klist=kklist)
            elif re.search('^Sym$',m.group(1)):
                kernel = Symmetric_Kernel(klist=kklist)
        else:
            if re.match('^C$',name):
                kernel = Constant_Kernel()
            elif re.match('^n$',name):
                kernel = Noise_Kernel()
            elif re.match('^L$',name):
                kernel = Linear_Kernel()
            elif re.match('^P$',name):
                kernel = Poly_Order_Kernel()
            elif re.match('^SE$',name):
                kernel = SE_Kernel()
            elif re.match('^RQ$',name):
                kernel = RQ_Kernel()
            elif re.match('^MH$',name):
                kernel = Matern_HI_Kernel()
            elif re.match('^NN$',name):
                kernel = NN_Kernel()
            elif re.match('^Gw',name):
                wname = re.search('^Gw(.*)$',name).group(1)
                wfunc = None
                if re.match('^C$',wname):
                    wfunc = Constant_WarpingFunction()
                elif re.match('^IG$',wname):
                    wfunc = IG_WarpingFunction()
                kernel = Gibbs_Kernel(wfunc=wfunc)
    return kernel


def KernelReconstructor(name,pars=None):
    """
    Function to reconstruct any :code:`_Kernel` instance from its codename and parameter list,
    useful for saving only necessary data to represent a :code:`GaussianProcessRregression1D`
    instance.

    .. note::

        All :code:`_OperatorKernel` class implementations should use encapsulating
        round brackets to specify their constituents. (v >= 1.0.1)

    :arg name: str. The codename of the desired :code:`_Kernel` instance.

    :kwarg pars: array. The hyperparameter and constant values to be stored in the :code:`_Kernel` instance, order determined by the specific :code:`_Kernel` class implementation. (optional)

    :returns: object. The desired :code:`_Kernel` instance, with the supplied parameters already set if parameters were valid. Returns :code:`None` if given kernel codename was invalid.
    """

    kernel = KernelConstructor(name)
    pvec = None
    if isinstance(pars,(list,tuple)):
        pvec = np.array(pars).flatten()
    elif isinstance(pars,np.ndarray):
        pvec = pars.flatten()
    if isinstance(kernel,_Kernel) and pvec is not None:
        nhyp = kernel.hyperparameters.size
        ncst = kernel.constants.size
        if ncst > 0 and pvec.size >= (nhyp + ncst):
            csts = pvec[nhyp:nhyp+ncst] if pvec.size > (nhyp + ncst) else pvec[nhyp:]
            kernel.constants = csts
        if pvec.size >= nhyp:
            theta = pvec[:nhyp] if pvec.size > nhyp else pvec.copy()
            kernel.hyperparameters = theta
    return kernel
