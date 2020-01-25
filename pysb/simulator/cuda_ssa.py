from __future__ import print_function
from pysb.simulator.base import SimulationResult
from pysb.simulator.ssa_base import SSABase
import numpy as np
import os
import time
import warnings
from pysb.pathfinder import get_path
try:
    import pycuda
    import pycuda.autoinit
    from pycuda.autoinit import device
    import pycuda as cuda
    import pycuda.compiler
    import pycuda.tools as tools
    import pycuda.driver as driver
    import pycuda.gpuarray as gpuarray
except ImportError:
    pycuda = None


class CudaSSASimulator(SSABase):
    """
    SSA simulator for NVIDIA gpus

    Use `CUDA_VISIBLE_DEVICES` to set the device.

    Requires `PyCUDA`_, `CUDA`_, and a CUDA compatible NVIDIA gpu.

    .. _PyCUDA :
        https://documen.tician.de/pycuda/
    .. _CUDA :
        https://developer.nvidia.com/cuda-zone

    Parameters
    ----------
    model : pysb.Model
        Model to simulate.
    tspan : vector-like, optional
        Time values over which to simulate. The first and last values define
        the time range. Returned trajectories are sampled at every value unless
        the simulation is interrupted for some reason, e.g., due to
        satisfaction
        of a logical stopping criterion (see 'tout' below).
    initials : vector-like or dict, optional
        Values to use for the initial condition of all species. Ordering is
        determined by the order of model.species. If not specified, initial
        conditions will be taken from model.initial_conditions (with
        initial condition parameter values taken from `param_values` if
        specified).
    param_values : vector-like or dict, optional
        Values to use for every parameter in the model. Ordering is
        determined by the order of model.parameters.
        If passed as a dictionary, keys must be parameter names.
        If not specified, parameter values will be taken directly from
        model.parameters.
    verbose : bool, optional (default: False)
        Verbose output.
    precision : (np.float64, np.float32)
        Precision for ssa simulation. Default is np.float64. float32 should
        be used with caution.

    Attributes
    ----------
    verbose: bool
        Verbosity flag passed to the constructor.
    model : pysb.Model
        Model passed to the constructor.
    tspan : vector-like
        Time values passed to the constructor.




    """
    _supports = {'multi_initials': True, 'multi_param_values': True}

    def __init__(self, model, verbose=False, tspan=None, precision=np.float64,
                 **kwargs):
        if pycuda is None:
            raise ImportError('pycuda library required for {}'
                              ''.format(self.__class__.__name__))
        super(CudaSSASimulator, self).__init__(model, verbose, **kwargs)
        self._device = device.name()
        self.tspan = tspan
        self.verbose = verbose
        # private attribute
        self._step_0 = True

        template_code = _load_template()
        self._code = template_code.format(**self._get_template_args())

        if precision not in (np.float64, np.float32):
            raise TypeError("CudaSSASimulator can only use np.float64 or "
                            "np.float32 precisions")
        self._dtype = precision
        if self._dtype == np.float32:
            self._code = self._code.replace('double', 'float')
            self._code = self._code.replace('USE_DP', 'USE_FLOAT')
            warnings.warn("Should be cautious using single precision")
        if verbose == 2:
            self._code = self._code.replace('//#define VERBOSE',
                                            '#define VERBOSE')
        elif verbose > 3:
            self._code = self._code.replace('//#define VERBOSE',
                                            '#define VERBOSE_MAX')

        self._ssa_all = None
        self._kernel = None
        self._param_tex = None
        self._ssa = None
        self._logger.info("Initialized CudaSSASimulator class")

    def _compile(self):

        if self.verbose:
            self._logger.info("Output cuda file to ssa_cuda_code.cu")
            with open("ssa_cuda_code.cu", "w") as source_file:
                source_file.write(self._code)
        nvcc_bin = get_path('nvcc')
        self._logger.debug("Compiling CUDA code")
        opts = ['-O3', '--use_fast_math']
        self._kernel = pycuda.compiler.SourceModule(
            self._code, nvcc=nvcc_bin, options=opts, no_extern_c=True,
        )

        self._ssa = self._kernel.get_function("Gillespie_all_steps")
        self._logger.debug("Compiled CUDA code")
        self._step_0 = False

    def run(self, tspan=None, param_values=None, initials=None, number_sim=0,
            threads_per_block=None):
        """
        Run a simulation and returns the result (trajectories)

        .. note::
            In early versions of the Simulator class, ``tspan``, ``initials``
            and ``param_values`` supplied to this method persisted to future
            :func:`run` calls. This is no longer the case.

        Parameters
        ----------
        tspan
        initials
        param_values
            See parameter definitions in :class:`ScipyOdeSimulator`.
        number_sim: int
            Number of simulations to perform
        threads_per_block: int
            Number of threads per block. Optimal value is generally 32

        Returns
        -------
        A :class:`SimulationResult` object
        """

        super(CudaSSASimulator, self).run(tspan=tspan, initials=initials,
                                          param_values=param_values,
                                          number_sim=number_sim)

        self._logger.info("Using device {}".format(device.name()))

        if tspan is None:
            tspan = self.tspan

        tout = [tspan] * self.num_sim
        t_out = np.array(tspan, dtype=self._dtype)

        # set default threads per block
        if threads_per_block is None:
            threads_per_block = 32

        blocks, threads = self.get_blocks(self.num_sim, threads_per_block)

        # compile kernel and send parameters to GPU
        if self._step_0:
            self._compile()
        if self.verbose:
            self._print_verbose(threads)

        #  Note, this number will be larger than n_simulations if the gpu grid
        #  is not filled. The rest will be filled with zeros and not impact
        #  results. They are trimmed right before passing to simulation results
        total_threads = int(blocks * threads)

        self._logger.info("Creating content on device")
        timer_start = time.time()
        param_array_gpu = gpuarray.to_gpu(
            self._create_gpu_array(self.param_values, total_threads,
                                   self._dtype)
        )

        species_matrix_gpu = gpuarray.to_gpu(
            self._create_gpu_array(self.initials, total_threads, np.int32)
        )

        # allocate and upload time to GPU
        time_points_gpu = gpuarray.to_gpu(np.array(t_out, dtype=self._dtype))

        # allocate space on GPU for results
        result = driver.managed_zeros(
            shape=(total_threads, len(t_out), self._n_species),
            dtype=np.int32, mem_flags=driver.mem_attach_flags.GLOBAL
        )
        elasped_t = time.time() - timer_start
        self._logger.info("Completed transfer in: {:.4f}s".format(elasped_t))

        self._logger.info("Starting {} simulations on {} blocks"
                          "".format(self.num_sim, blocks))

        timer_start = time.time()
        # perform simulation
        self._ssa(species_matrix_gpu, result, time_points_gpu,
                  np.int32(len(t_out)), param_array_gpu,
                  block=(threads, 1, 1), grid=(blocks, 1))

        # Wait for kernel completion before host access
        pycuda.autoinit.context.synchronize()

        self._time = time.time() - timer_start
        self._logger.info("{} simulations "
                          "in {:.4f}s".format(self.num_sim, self._time))

        # retrieve and store results, only keeping num_sim (desired quantity)
        return SimulationResult(self, tout, result[:self.num_sim, :, :])

    def _print_verbose(self, threads):
        # Beyond this point is just pretty printing
        self._logger.debug("Attributes for device {}".format(device.name()))
        for (key, value) in device.get_attributes().items():
            self._logger.debug("{}:{}".format(key, value))
        self._logger.debug("threads = {}".format(threads))
        kern = self._ssa
        self._logger.debug("Local memory  = {}".format(kern.local_size_bytes))
        self._logger.debug("Shared memory = {}".format(kern.shared_size_bytes))
        self._logger.debug("Registers  = {}".format(kern.num_regs))

        occ = tools.OccupancyRecord(tools.DeviceData(),
                                    threads=threads,
                                    shared_mem=kern.shared_size_bytes,
                                    registers=kern.num_regs)
        self._logger.debug("tb_per_mp  = {}".format(occ.tb_per_mp))
        self._logger.debug("limited by = {}".format(occ.limited_by))
        self._logger.debug("occupancy  = {}".format(occ.occupancy))
        self._logger.debug("tb/mp limits  = {}".format(occ.tb_per_mp_limits))

    @staticmethod
    def _create_gpu_array(values, total_threads, prec):

        # Create species matrix on GPU
        # will make according to number of total threads, not n_simulations
        gpu_array = np.zeros((total_threads, values.shape[1]), dtype=prec)
        # Filling species matrix
        # Note that this might not fill entire array that was created.
        # The rest of the array will be zeros to fill up GPU.
        gpu_array[:len(values)] = values
        return gpu_array

    def get_blocks(self, n_simulations, threads_per_block):
        max_tpb = 256
        if threads_per_block > max_tpb:
            self._logger.warning("Limit of 256 threads per block due to curand."
                            " Setting to 256.")
            threads_per_block = max_tpb
        if n_simulations < max_tpb:
            block_count = 1
            threads_per_block = max_tpb
        elif n_simulations % threads_per_block == 0:
            block_count = int(n_simulations // threads_per_block)
        else:
            block_count = int(n_simulations // threads_per_block + 1)
        return block_count, threads_per_block


def _load_template():
    _path = os.path.join(os.path.dirname(__file__), 'templates', 'ssa.cu')
    with open(_path, 'r') as f:
        gillespie_code = f.read()
    return gillespie_code

