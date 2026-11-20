.. _ref_references:

References
**********

To cite ToMoBAR, use the conference paper [CT2020]_. Linked titles below lead
to the published articles through their DOI; the conference PDF is bundled
with this documentation.

Software and reconstruction infrastructure
------------------------------------------

The Savu paper [SX2022]_ describes large-scale synchrotron reconstruction.
CCPi-Regularisation-Toolkit [KAZ2019]_ provides historical algorithm and software
context; the current native regularisers do not require that external toolkit.

.. [CT2020] D. Kazantsev and N. Wadeson (2020).
   Tomographic MOdel-BAsed Reconstruction (ToMoBAR) software for high resolution
   synchrotron X-ray tomography. *CT Meeting 2020*.
   :download:`Conference paper PDF <../../Kazantsev_CT_20.pdf>`.

.. [VanAarle2015] W. van Aarle et al. (2015).
   `The ASTRA Toolbox: A platform for advanced algorithm development in electron
   tomography <https://doi.org/10.1016/j.ultramic.2015.05.002>`_.
   *Ultramicroscopy*, 157, 35–47.

.. [SX2022] D. Kazantsev, N. Wadeson and M. Basham (2022).
   `High performance Savu software for fast 3D model-based iterative reconstruction
   of large data at Diamond Light Source <https://doi.org/10.1016/j.softx.2022.101157>`_.
   *SoftwareX*, 19, 101157.

.. [KAZ2019] D. Kazantsev, E. Pasca, M. J. Turner and P. J. Withers (2019).
   `CCPi-Regularisation toolkit for computed tomographic image reconstruction
   with proximal splitting algorithms <https://doi.org/10.1016/j.softx.2019.04.003>`_.
   *SoftwareX*, 9, 317–323.

.. [NIKITIN2023] V. Nikitin (2023).
   `TomocuPy – efficient GPU-based tomographic reconstruction with asynchronous
   data processing <https://doi.org/10.1107/S1600577522010311>`_.
   *Journal of Synchrotron Radiation*, 30, 179–191.

Iterative and Fourier methods
-----------------------------

.. [BT2009] A. Beck and M. Teboulle (2009).
   `A fast iterative shrinkage-thresholding algorithm for linear inverse problems
   <https://doi.org/10.1137/080716542>`_.
   *SIAM Journal on Imaging Sciences*, 2(1), 183–202.

.. [Boyd2011] S. Boyd, N. Parikh, E. Chu, B. Peleato and J. Eckstein (2011).
   `Distributed optimization and statistical learning via the alternating
   direction method of multipliers <https://doi.org/10.1561/2200000016>`_.
   *Foundations and Trends in Machine Learning*, 3(1), 1–122.

.. [Xu2016] Q. Xu et al. (2016).
   `Accelerated fast iterative shrinkage thresholding algorithms for
   sparsity-regularized cone-beam CT image reconstruction
   <https://doi.org/10.1118/1.4942812>`_.
   *Medical Physics*, 43(4), 1849–1872.

.. [HL1994] H. M. Hudson and R. S. Larkin (1994).
   `Accelerated image reconstruction using ordered subsets of projection data
   <https://doi.org/10.1109/42.363108>`_.
   *IEEE Transactions on Medical Imaging*, 13(4), 601–609.

.. [DR1993] A. Dutt and V. Rokhlin (1993).
   `Fast Fourier transforms for nonequispaced data <https://doi.org/10.1137/0914081>`_.
   *SIAM Journal on Scientific Computing*, 14(6), 1368–1393.

.. [NIKITIN2017] V. V. Nikitin et al. (2017).
   `Fast hyperbolic Radon transform represented as convolutions in log-polar
   coordinates <https://doi.org/10.1016/j.cageo.2017.04.013>`_.
   *Computers & Geosciences*, 105, 21–33.

Regularisation
---------------

.. [ROF1992] L. I. Rudin, S. Osher and E. Fatemi (1992).
   `Nonlinear total variation based noise removal algorithms
   <https://doi.org/10.1016/0167-2789(92)90242-F>`_.
   *Physica D: Nonlinear Phenomena*, 60(1–4), 259–268.

.. [CP2011] A. Chambolle and T. Pock (2011).
   `A first-order primal-dual algorithm for convex problems with applications
   to imaging <https://doi.org/10.1007/s10851-010-0251-1>`_.
   *Journal of Mathematical Imaging and Vision*, 40, 120–145.

.. [GO2009] T. Goldstein and S. Osher (2009).
   `The split Bregman method for L1-regularized problems
   <https://doi.org/10.1137/080725891>`_.
   *SIAM Journal on Imaging Sciences*, 2(2), 323–343.

.. [FGP2009] A. Beck and M. Teboulle (2009).
   `Fast gradient-based algorithms for constrained total variation image
   denoising and deblurring problems <https://doi.org/10.1109/TIP.2009.2028250>`_.
   *IEEE Transactions on Image Processing*, 18(11), 2419–2434.

.. [TGV2010] K. Bredies, K. Kunisch and T. Pock (2010).
   `Total generalized variation <https://doi.org/10.1137/090769521>`_.
   *SIAM Journal on Imaging Sciences*, 3(3), 492–526.

.. [PM1990] P. Perona and J. Malik (1990).
   `Scale-space and edge detection using anisotropic diffusion
   <https://doi.org/10.1109/34.56205>`_.
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(7), 629–639.

Data models and experimental datasets
-------------------------------------

The robust Student's t model [KAZ1_2017]_ and Group-Huber ring-correction work
[PM2015]_ are historical background, rather than selectable fidelity options
in the current API.

.. [HOA2017] H. O. Aggrawal, M. S. Andersen, S. Rose and E. Y. Sidky (2018).
   `A convex reconstruction model for X-ray tomographic imaging with uncertain
   flat-fields <https://doi.org/10.1109/TCI.2017.2723246>`_.
   *IEEE Transactions on Computational Imaging*, 4(1), 17–31.
   First published online in 2017.

.. [PM2015] P. Paleo and A. Mirone (2015).
   `Ring artifacts correction in compressed sensing tomographic reconstruction
   <https://doi.org/10.1107/S1600577515010176>`_.
   *Journal of Synchrotron Radiation*, 22(5), 1268–1278.

.. [KAZ1_2017] D. Kazantsev et al. (2017).
   `A novel tomographic reconstruction method based on the robust Student's t
   function for suppressing data outliers <https://doi.org/10.1109/TCI.2017.2694607>`_.
   *IEEE Transactions on Computational Imaging*, 3(4), 682–693.

.. [GUO2018] E. Guo et al. (2018).
   `The influence of nanoparticles on dendritic grain growth in Mg alloys
   <https://doi.org/10.1016/j.actamat.2018.04.023>`_.
   *Acta Materialia*.

.. [KAZ2017] D. Kazantsev et al. (2017).
   `Model-based iterative reconstruction using higher-order regularization of
   dynamic synchrotron data <https://doi.org/10.1088/1361-6501/aa7fa8>`_.
   *Measurement Science and Technology*, 28(9), 094004.

Geometry and alignment
----------------------

.. [HELFEN2007] L. Helfen et al. (2007).
   `Synchrotron-radiation computed laminography for high-resolution
   three-dimensional imaging of flat devices <https://doi.org/10.1002/pssa.200775676>`_.
   *Physica Status Solidi (a)*, 204, 2760–2765.

.. [GURSOY2017] D. Gürsoy et al. (2017).
   `Rapid alignment of nanotomography data using joint iterative reconstruction
   and reprojection <https://doi.org/10.1038/s41598-017-12141-9>`_.
   *Scientific Reports*, 7, 11818.
