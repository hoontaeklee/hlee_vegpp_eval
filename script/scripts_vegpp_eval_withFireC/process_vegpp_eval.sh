#!/bin/bash

#-------- set environment
path_python="/Net/Groups/BGI/people/hlee/anaconda3/envs/data_p/bin/python"
path_sindbad_output="/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712"
#-------- process non-sindbad data sets
# an array of scripts to run
# ar_pypath=(
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_detrend_esv-iav.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_calc_regional_msciav_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_calc_regional_msciav_koeppengeiger5ns.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_calc_regional_msciav_transcom.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_calc_cov_matrix_by_regions_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_fluxcom_calc_cov_matrix_by_regions_koeppengeiger5ns.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_detrend_esv-iav.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_calc_regional_msciav_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_calc_regional_msciav_koeppengeiger5ns.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_calc_regional_msciav_transcom.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_calc_cov_matrix_by_regions_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_oco2mip_calc_cov_matrix_by_regions_koeppengeiger5ns.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_detrend_v9.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_calc_regional_msciav_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_calc_regional_msciav_koeppengeiger5ns.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_calc_regional_msciav_transcom.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_calc_cov_matrix_by_regions_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/process_trendy_calc_cov_matrix_by_regions_koeppengeiger5ns.py
# )

# for p in ${ar_pypath[@]}; do
#     printf "running $p \n"
#     # chmod 755 $p
#     $path_python $p
# done
#--------

#-------- process the reformatted sindbad output - run with OCO-2 MIP  as NEE constraint
# an array of scripts to run
# ar_pypath=(
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/calc_esv_iav_sindbad.py
# )
# for p in ${ar_pypath[@]}; do
#     printf "running $p \n"
#     # chmod 755 $p
#     $path_python $p $path_sindbad_output
# done
#--------

#-------- plot detrended variables
ar_pypath=(
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/calc_vegpp_cov_matrix_by_regions_koeppengeiger5.py
# /Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/calc_vegpp_cov_matrix_by_regions_koeppengeiger5ns.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_fig05_nee_msc_by_regions.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_fig06_nee_iav_by_regions.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_fig08_msc_contrib_by_regions_koeppengeiger5.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_fig09_iav_contrib_by_regions_koeppengeiger5.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_figa11_msc_contrib_by_regions_koeppengeiger5ns.py
/Net/Groups/BGI/people/hlee/scripts/scripts_vegpp_eval_withFireC/plot_figa12_iav_contrib_by_regions_koeppengeiger5ns.py
)

path_sindbad_detrended_output="${path_sindbad_output}/detrended"
for p in ${ar_pypath[@]}; do
    printf "running $p \n"
    # chmod 755 $p
    $path_python $p $path_sindbad_detrended_output
done
#--------

printf "Done! \n"

exit