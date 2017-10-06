import numpy as np
import matplotlib.pyplot as plt

def plot_func(log_dir, iter_id, loss_list, local_curv_list, max_curv_list, min_curv_list,
             lr_g_norm_list, lr_g_norm_squared_list, lr_list, dr_list, mu_list, grad_avg_norm_list,
             dist_list, grad_var_list, move_lr_g_norm_list, move_lr_g_norm_squared_list, 
             fast_view_act_list, lr_grad_norm_clamp_act_list):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
    plt.figure()
    plt.semilogy(loss_list, '.', alpha=0.2, label="Loss")
    plt.semilogy(running_mean(loss_list,100), label="Average Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    ax = plt.subplot(111)
    plt.legend(ncol=3, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_loss_iter_" + str(iter_id) + ".pdf")
    plt.close()

    plt.figure()
    plt.semilogy(lr_g_norm_list, label="lr * grad norm")
    plt.semilogy(local_curv_list, label="local curvature")
    plt.semilogy(max_curv_list, label="max curv in win")
    plt.semilogy(min_curv_list, label="min curv in win")
#         plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
    #plt.semilogy(lr_g_norm_list, label="lr * grad norm")
   # plt.semilogy(lr_g_norm_squared_list, label="lr * grad norm squared")
   # plt.semilogy(move_lr_g_norm_list, label="lr * grad norm move")
   # plt.semilogy(move_lr_g_norm_squared_list, label="lr * grad norm squared move")
    if np.min(lr_g_norm_list) < 1e-9 or np.min(local_curv_list) < 1e-9 or np.min(max_curv_list) < 1e-9 or np.min(min_curv_list) < 1e-9:
      plt.ylim([1e-9, None] )

    plt.title("On local curvature")
    plt.grid()
    ax = plt.subplot(111)
    ax.legend(ncol=2, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_curv_iter_" + str(iter_id) + ".pdf")
    plt.close()

    plt.figure()
    #plt.semilogy(local_curv_list, label="local curvature")
    #plt.semilogy(max_curv_list, label="max curv in win")
    #plt.semilogy(min_curv_list, label="min curv in win")
#         plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
    plt.semilogy(lr_g_norm_list, label="lr * grad norm")
    #plt.semilogy(lr_g_norm_squared_list, label="lr * grad norm squared")
    plt.semilogy(move_lr_g_norm_list, label="lr * grad norm move")
    #plt.semilogy(move_lr_g_norm_squared_list, label="lr * grad norm squared move")
    if np.min(lr_g_norm_list) < 1e-9 or np.min(move_lr_g_norm_list) < 1e-9:
      plt.ylim([1e-9, None] )
    plt.title("On local curvature")
    plt.grid()
    ax = plt.subplot(111)
    ax.legend(ncol=2, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_move_iter_" + str(iter_id) + ".pdf")
    plt.close()

    plt.figure()
    #plt.semilogy(local_curv_list, label="local curvature")
    #plt.semilogy(max_curv_list, label="max curv in win")
    #plt.semilogy(min_curv_list, label="min curv in win")
#         plt.semilogy(clip_norm_base_list, label="Clipping Thresh.")
    plt.semilogy(lr_g_norm_squared_list, label="lr * grad norm squared")
    plt.semilogy(move_lr_g_norm_squared_list, label="lr * grad norm squared move")
    if np.min(lr_g_norm_squared_list) < 1e-9 or np.min(move_lr_g_norm_squared_list) < 1e-9:
      plt.ylim([1e-9, None] )
    plt.title("On local curvature")
    plt.grid()
    ax = plt.subplot(111)
    ax.legend(ncol=2, fancybox=True, shadow=True)
    plt.savefig(log_dir + "/fig_move_squared_iter_" + str(iter_id) + ".pdf")
    plt.close()

    plt.figure()
    plt.semilogy(lr_list, label="lr min")
    plt.semilogy(dr_list, label="dynamic range")
    plt.semilogy(mu_list, label="mu")
    plt.semilogy(grad_avg_norm_list, label="Grad avg norm")
    plt.semilogy(dist_list, label="Est dist from opt")
    plt.semilogy(grad_var_list, label="Grad variance")
    plt.semilogy(fast_view_act_list, label="fast_view_act lr")
    plt.semilogy(lr_grad_norm_clamp_act_list, label="lr grad norm clamp lr")
    if np.min(grad_var_list) < 1e-9 or np.min(fast_view_act_list) < 1e-9 or np.min(lr_grad_norm_clamp_act_list) < 1e-9:
      plt.ylim([1e-9, None] )
    plt.title('LR='+str(lr_list[-1])+' mu='+str(mu_list[-1] ) )
    plt.grid()
    plt.legend()
    plt.savefig(log_dir + "/fig_hyper_iter_" + str(iter_id) + ".pdf")
    plt.close()
