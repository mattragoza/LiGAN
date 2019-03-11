
python results.py expts/LOSSWEIGHT0_CE -o loss_weight_results/LW01_CE -f 0 -i 100000 --plot_lines --n_cols 6  \
	--hue model_name --avg_iters 100 \
	-y disc_log_loss        --ylim '[-0.1, 1.1]' \
	-y gen_adv_log_loss     --ylim '[-4, 44]' \
	-y gen_L2_loss          --ylim '[-100, 1100]' \
	-y disc_grad_norm       --ylim '[-40, 440]' \
	-y gen_grad_norm        --ylim '[-40, 440]' \
	-y gen_loss_weight      --ylim '[-0.1, 1.1]' \
	-y lig_norm             --ylim '[-4, 44]' \
	-y lig_gen_norm         --ylim '[-4, 44]' \
	-y lig_lig_gen_dist     --ylim '[-4, 44]' \
	-y lig_lig_dist         --ylim '[-4, 44]' \
	-y lig_gen_lig_gen_dist --ylim '[-4, 44]' 

python results.py expts/LOSSWEIGHT0_AE -o loss_weight_results/LW01_AE -f 0 -i 100000 --plot_lines --n_cols 6  \
	--hue model_name --avg_iters 100 \
	-y disc_log_loss        --ylim '[-0.1, 1.1]' \
	-y gen_adv_log_loss     --ylim '[-4, 44]' \
	-y gen_L2_loss          --ylim '[-100, 1100]' \
	-y disc_grad_norm       --ylim '[-40, 440]' \
	-y gen_grad_norm        --ylim '[-40, 440]' \
	-y gen_loss_weight      --ylim '[-0.1, 1.1]' \
	-y lig_norm             --ylim '[-4, 44]' \
	-y lig_gen_norm         --ylim '[-4, 44]' \
	-y lig_lig_gen_dist     --ylim '[-4, 44]' \
	-y lig_lig_dist         --ylim '[-4, 44]' \
	-y lig_gen_lig_gen_dist --ylim '[-4, 44]' 

