# Essential tests
This is a curated entry point for routine development and human review. It
selects existing tests; it does not replace the full regression suite or remove
any cases. The `essential` decorators beside the test functions are the source
of truth for selection. All parametrisations of a marked function run, including
fixture parametrisations, so one readable test may produce several cases.

The initial selection contains **40 test functions across 21 files**, expanding
to **192 cases** in the development environment with optional dependencies
installed. The verified run with GPU access passed 189 cases; three existing
unsupported solver/subset combinations skipped. The CPU-only run passed 115
cases and skipped 77 (74 GPU cases plus those three unsupported combinations).
The remaining 4,488 collected cases are deselected only when using the marker.
Counts can change as features and parameter combinations evolve.

## Run the suite

Run from the repository root with the ToMoBAR environment activated:

```sh
python -m pytest -m essential                   # Essential CPU and GPU cases
python -m pytest -m essential --cpu-only        # CPU portion only
python -m pytest -m "essential and gpu"          # GPU portion only
python -m pytest -m essential --collect-only -q # Inspect the selected cases
```

GPU cases skip when CUDA is unavailable. Check the skip summary: a CPU-only
success does not validate ASTRA or CUDA kernels. Matplotlib plot tests also
require the optional visualisation dependency. The CPU Fourier reference tests
currently require CuPy to be installed, although they perform their computation
with NumPy. That limitation is inherited from the existing tests.

The usual `python -m pytest` command still runs all ordinary tests. Use it before
a release and for changes to solver numerics, projection backends, CUDA kernels,
array layouts or resource ownership. Existing `--full` and `--performance` suites
remain separate opt-in checks; no expensive cases are added to this selection.

## How to read and maintain this selection

For a short first reading, start with these five examples in the table below:
`test_geometry_preserves_axis_and_view_order`, `test_tv_energy`,
`test_mlem_two_voxel_regression`,
`test_zero_boundary_flux_has_analytic_solution`, and
`test_known_phantom_has_consistent_physical_scale`. They progress from input
conventions and small hand-checkable results to real GPU reconstruction.

Start with the geometry and small analytical examples, then read the GPU
integration tests. For each test, identify its input, the public behaviour or
mathematical result it protects, and the reason for its tolerance. Some selected
tests use a small NumPy solver host or an isolated backend double; the GPU cases
complement those checks by exercising real CUDA implementations.

The selection favours meaningful numerical references and observable contracts.
It deliberately leaves exhaustive validation matrices, allocation/performance
probes, most precision/layout combinations, and detailed implementation checks
in the broader regression suite. It is not exhaustive coverage of every filter
or regulariser: ROF-TV and FGP-TV, for example, remain in the broader suite.

When adding or changing a feature, first check whether one of these tests can
express its essential behaviour clearly. Add the marker only for a distinct,
important guarantee; keep detailed edge cases in the regression suite. Retain
previous bug regressions even when they are outside this selection. Do not
refresh numerical expectations just to make a failing test pass.

The table below explains every selected test. If a marked test is renamed,
removed or replaced, update its entry here. No duplicate test bodies or custom
selection machinery are needed.

## Selected behaviours

### `core/test_geometry.py`

| Test | Why it belongs |
| --- | --- |
| [`test_geometry_preserves_axis_and_view_order`](core/test_geometry.py) | Geometry retains the supplied axes and acquisition order. |
| [`test_cor_correction_does_not_populate_detector_shifts`](core/test_geometry.py) | Centre-of-rotation correction remains separate from detector motion. |
| [`test_geometry_rejects_invalid_angles`](core/test_geometry.py) | Empty, nonfinite and malformed angle arrays are rejected. |
| [`test_view_subsets_are_backend_independent_immutable_partitions`](core/test_geometry.py) | Subsets partition every acquired view without changing geometry. |

### `core/test_array_prepare.py`

| Test | Why it belongs |
| --- | --- |
| [`test_numpy_axes_and_padding_without_input_mutation`](core/test_array_prepare.py) | Axis conversion and detector padding preserve caller data. |
| [`test_single_detector_row_and_invalid_shapes`](core/test_array_prepare.py) | Single-row inputs work and incompatible shapes fail clearly. |
| [`test_cgls_preserves_projection_input`](core/test_array_prepare.py) | A real GPU CGLS reconstruction preserves measured projections. |

### `core/test_architecture.py`

| Test | Why it belongs |
| --- | --- |
| [`test_metadata_state_and_kernel_package_work_without_gpu_imports`](core/test_architecture.py) | Geometry, settings and inspection metadata import without CUDA. |

### `core/test_processing.py`

| Test | Why it belongs |
| --- | --- |
| [`test_processing_needs_only_numpy`](core/test_processing.py) | CPU preprocessing remains usable without GPU dependencies. |
| [`test_explicit_postprocessing_preserves_mask_behavior`](core/test_processing.py) | Output masking follows the documented numerical convention. |

### `core/test_visualisation.py`

| Test | Why it belongs |
| --- | --- |
| [`test_volume_zyx_bounds_and_odd_detector_pixel_edges`](core/test_visualisation.py) | Geometry plots use correct volume bounds and detector edges. |

### `reconstruction/test_projection_operator.py`

| Test | Why it belongs |
| --- | --- |
| [`test_lazy_shared_projector_and_output_buffer_identity`](reconstruction/test_projection_operator.py) | Projection operations reuse owned resources and supplied output buffers. |
| [`test_context_failure_and_closed_calls_release_resources`](reconstruction/test_projection_operator.py) | Resources are released after exceptions and closed objects reject calls. |

### `reconstruction/test_backend_selection.py`

| Test | Why it belongs |
| --- | --- |
| [`test_unsupported_geometry_rejected_without_cuda`](reconstruction/test_backend_selection.py) | Unsupported Fourier geometries fail before CUDA resources are allocated. |

### `astra/test_integration.py`

| Test | Why it belongs |
| --- | --- |
| [`test_direct_full_and_subset_projection_on_cuda`](astra/test_integration.py) | Real ASTRA forward/backprojection agrees across full and subset operations. |
| [`test_iterative_output_crop_matches_full_working_volume`](astra/test_integration.py) | Public iterative output cropping matches a full-grid reconstruction. |

### `reconstruction/test_quantitative_scaling.py`

| Test | Why it belongs |
| --- | --- |
| [`test_known_phantom_has_consistent_physical_scale`](reconstruction/test_quantitative_scaling.py) | FBP, Fourier inversion and CGLS recover a known phantom amplitude. |

### `solvers/test_solver_state.py`

| Test | Why it belongs |
| --- | --- |
| [`test_batches_equal_uninterrupted_and_oneshot`](solvers/test_solver_state.py) | All six solvers give the same result when iterations are split into batches. |
| [`test_geometry_restart_matches_fresh_warm_start`](solvers/test_solver_state.py) | Geometry changes restart solver work consistently with a fresh warm start. |
| [`test_owns_data_and_reuses_preparation_and_norm`](solvers/test_solver_state.py) | Continuation owns its measured data and avoids repeated preparation. |
| [`test_cuda_continuation`](solvers/test_solver_state.py) | Continuation agrees with one-shot execution on real GPU reconstructions. |

### `solvers/test_admm.py`

| Test | Why it belongs |
| --- | --- |
| [`test_first_step_uses_relaxed_data_for_prox_and_dual`](solvers/test_admm.py) | A two-pixel example checks the ADMM primal and dual update equations. |
| [`test_cg_data_solve_matches_dense_quadratic_system`](solvers/test_admm.py) | ADMM CG agrees with an independently solved small linear system. |

### `solvers/test_osem_normalization.py`

| Test | Why it belongs |
| --- | --- |
| [`test_mlem_two_voxel_regression`](solvers/test_osem_normalization.py) | A two-voxel calculation checks expectation-maximisation normalisation. |

### `solvers/test_solver_options.py`

| Test | Why it belongs |
| --- | --- |
| [`test_unused_options_fail_before_preparing_arrays`](solvers/test_solver_options.py) | Unsupported solver options fail before array preparation. |
| [`test_public_signatures_match_option_schema_and_defaults`](solvers/test_solver_options.py) | Public parameter names and defaults match persistent solver settings. |
| [`test_common_parameters_validated_without_reconstructor`](solvers/test_solver_options.py) | Shared solver settings reject incorrect types. |

### `solvers/test_energy.py`

| Test | Why it belongs |
| --- | --- |
| [`test_fidelity_gradient`](solvers/test_energy.py) | LS, PWLS and SWLS gradients match finite differences of their objectives. |
| [`test_tv_energy`](solvers/test_energy.py) | TV energy matches a hand-checkable two-by-two image. |

### `fourier/test_usfft_numerics.py`

| Test | Why it belongs |
| --- | --- |
| [`test_matches_direct_plane_wave_reference`](fourier/test_usfft_numerics.py) | USFFT projections agree with independent direct Fourier summation. |
| [`test_cardinal_angles_orientation_mass_and_angle_count`](fourier/test_usfft_numerics.py) | Cardinal projections preserve orientation, mass and angular scaling. |

### `solvers/test_usfft_integration.py`

| Test | Why it belongs |
| --- | --- |
| [`test_supported_solvers_reduce_residual`](solvers/test_usfft_integration.py) | Every USFFT-supported solver reduces the data residual on a small phantom. |

### `regularisers/test_pd_tv.py`

| Test | Why it belongs |
| --- | --- |
| [`test_zero_boundary_flux_has_analytic_solution`](regularisers/test_pd_tv.py) | PD-TV agrees with an analytical two-value denoising solution. |
| [`test_zero_strength_is_identity_or_nonnegative_projection`](regularisers/test_pd_tv.py) | Zero-strength TV preserves the image or applies only nonnegativity. |

### `regularisers/test_sb_tv.py`

| Test | Why it belongs |
| --- | --- |
| [`test_sb_solves_tv_objective_at_boundaries`](regularisers/test_sb_tv.py) | Split Bregman solves a small TV problem with the expected boundary behaviour. |

### `regularisers/test_ndf.py`

| Test | Why it belongs |
| --- | --- |
| [`test_validation_and_zero_strength`](regularisers/test_ndf.py) | Diffusion validates settings and preserves the image at zero strength. |

### `regularisers/test_tgv.py`

| Test | Why it belongs |
| --- | --- |
| [`test_one_iteration_returns_true_primal_and_safe_step`](regularisers/test_tgv.py) | A small TGV update checks the returned primal image and step size. |

### `alignment/test_alignment.py`

| Test | Why it belongs |
| --- | --- |
| [`test_fractional_translation`](alignment/test_alignment.py) | Registration recovers a known fractional translation. |
| [`test_alignment_geometry_updates_and_final_state`](alignment/test_alignment.py) | Alignment recovers known shifts and updates the final geometry consistently. |
| [`test_failure_after_geometry_update_rolls_back`](alignment/test_alignment.py) | A failed alignment restores the original geometry. |
