#!/bin/bash
measurement_report_directory="./measurement_reports"
mkdir -p $measurement_report_directory

# regulariser="PD_TV"
# regulariser="PD_TV_fused"
regulariser="PD_TV_separate_p_fused"
if [ "$regulariser" = "PD_TV" ]; then
  kernels_to_measure=('dualPD3D_kernel' 'Proj_funcPD3D_iso_kernel' 'Proj_funcPD3D_aniso_kernel' 'DivProj3D_kernel' 'PDnonneg3D_kernel' 'getU3D_kernel')
elif [ "$regulariser" = "PD_TV_fused" ]; then
  kernels_to_measure=('primal_dual_for_total_variation_3D')
elif [ "$regulariser" = "PD_TV_separate_p_fused" ]; then
  kernels_to_measure=('primal_dual_for_total_variation_3D' 'dualPD3D_kernel')
fi

# for i in {1024,2048,4096,5120,6144}; do
#   for j in "${block_dims_for_center[@]}"; do
    report_filename="${measurement_report_directory}/report_${regulariser}.nsys-rep"
    output=$(sudo env "PATH=$PATH" nsys profile \
      --trace=cuda,cublas,nvtx \
      --python-sampling=true \
      --cuda-memory-usage=true \
      --force-overwrite=true \
      --output $report_filename \
      python measurement_target.py --regulariser $regulariser --measurement_target "regularisation")

    dict_line=$(echo "$output" | sed -n "/{.*runtime_ms.*}/p")
    cleaned_dict_line=$(echo "$dict_line" | sed "s/[{}']//g")

    end_to_end_runtime_ms=$(echo "$cleaned_dict_line" | sed -n "s/.*end_to_end_runtime_ms: \([0-9.]*\).*/\1/p")
    target_runtime_ms=$(echo "$cleaned_dict_line" | sed -n "s/.*target_runtime_ms: \([0-9.]*\).*/\1/p")
    measurement_iteration_count=$(echo "$cleaned_dict_line" | sed -n "s/.*measurement_iteration_count: \([0-9]*\).*/\1/p")

    sqlite_filename=${report_filename/.nsys-rep/.sqlite}
    nsys export --type sqlite --force-overwrite=true --output $sqlite_filename $report_filename

    python3 parse_sqlite.py \
      --db_file "$sqlite_filename" \
      --target_kernels "${kernels_to_measure[@]}" \
      --end_to_end_runtime_ms "$end_to_end_runtime_ms" \
      --target_runtime_ms "$target_runtime_ms" \
      --measurement_iteration_count "$measurement_iteration_count"
#   done
# done
