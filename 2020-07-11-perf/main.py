#2020-07-11-auroc-auprc.py

import os
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt

def add_true_positives(y_true, y_score, n, decision_thresh):
    """Return <y_true> and <y_score> so that they contain <n> additional
    true positives assuming a decision threshold of <decision_thresh>.
    True positives are predicted positive and actually positive."""
    #Predicted positive:
    pos_scores = np.random.uniform(low = decision_thresh+0.001, high = 1.0, size = n)
    y_score_new = y_score+pos_scores.tolist()
    #Actually positive:
    y_true_new = y_true+([1]*n)
    return y_true_new, y_score_new
    
def add_false_positives(y_true, y_score, n, decision_thresh):
    """Return <y_true> and <y_score> so that they contain <n> additional
    false positives assuming a decision threshold of <decision_thresh>.
    False positives are predicted positive and actually negative."""
    #Predicted positive:
    pos_scores = np.random.uniform(low = decision_thresh+0.001, high = 1.0, size = n)
    y_score_new = y_score+pos_scores.tolist()
    #Actually negative:
    y_true_new = y_true+([0]*n)
    return y_true_new, y_score_new
    
def add_true_negatives(y_true, y_score, n, decision_thresh):
    """Return <y_true> and <y_score> so that they contain <n> additional
    true negatives assuming a decision threshold of <decision_thresh>.
    True negatives are predicted negative and actually negative."""
    #Predicted negative:
    neg_scores = np.random.uniform(low = 0, high = decision_thresh-0.001, size = n)
    y_score_new = y_score+neg_scores.tolist()
    #Actually negative:
    y_true_new = y_true+([0]*n)
    return y_true_new, y_score_new

def add_false_negatives(y_true, y_score, n, decision_thresh):
    """Return <y_true> and <y_score> so that they contain <n> additional
    false negatives assuming a decision threshold of <decision_thresh>.
    False negatives are predicted negative and actually positive."""
    #Predicted negative:
    neg_scores = np.random.uniform(low = 0, high = decision_thresh-0.001, size = n)
    y_score_new = y_score+neg_scores.tolist()
    #Actually positive:
    y_true_new = y_true+([1]*n)
    return y_true_new, y_score_new


def confusion_matrix_string(y_true, y_score, decision_thresh):
    """Return the confusion matrix"""
    #Obtain binary predicted labels by applying <decision_thresh> to <y_score>
    y_pred = (np.array(y_score) > decision_thresh)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    #TPR_recall = float(true_pos)/(true_pos + false_neg)
    #FPR = float(false_pos)/(false_pos+true_neg)
    #precision = float(true_pos)/(true_pos + false_pos)
    return 'tn='+str(true_neg)+', fp='+str(false_pos)+', fn='+str(false_neg)+', tp='+str(true_pos)

class PlotAll(object):
    def __init__(self,title,savetitle,y_true,y_score):
        self.y_true = y_true
        self.y_score = y_score
        
        #Syle
        self.roc_color = 'crimson'
        self.pr_color = 'royalblue'
        self.main_linestyle = 'solid'
        self.neutral_color = 'k'
        self.neutral_linestyle = 'dashed'
        self.lw = 2
        
        #https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib
        #https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/figure_title.html
        fig, self.ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8,4.5))
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        #Matplotlib tight layout doesn't take into account title so we pass
        #the rect argument
        #https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join('figures',savetitle+'.png'))

    def plot_roc_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
        fpr, tpr_recall, _ = sklearn.metrics.roc_curve(self.y_true,self.y_score,pos_label = 1)
        roc_curve, = self.ax[0].plot(fpr, tpr_recall, color=self.roc_color, lw=self.lw, linestyle = self.main_linestyle)
        self.ax[0].fill_between(fpr, tpr_recall, step='post', alpha=0.2, color=self.roc_color)
        self.ax[0].plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle) #diagonal line
        self.ax[0].set_xlim([0.0, 1.0])
        self.ax[0].set_ylim([0.0, 1.05])
        self.ax[0].set_xlabel('False Positive Rate')
        self.ax[0].set_ylabel('True Positive Rate (Recall)')
        self.ax[0].set_title('AUROC=%0.2f' % sklearn.metrics.auc(fpr, tpr_recall))
    
    def plot_precision_recall_curve(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        precision, tpr_recall, _ = sklearn.metrics.precision_recall_curve(self.y_true, self.y_score)
        pr_curve, = self.ax[1].step(tpr_recall, precision, color=self.pr_color, alpha=0.2, where='post',linewidth=self.lw,linestyle=self.main_linestyle)
        self.ax[1].fill_between(tpr_recall, precision, step='post', alpha=0.2, color=self.pr_color)
        self.ax[1].set_xlabel('True Positive Rate (Recall)')
        self.ax[1].set_ylabel('Precision')
        self.ax[1].set_ylim([0.0, 1.05])
        self.ax[1].set_xlim([0.0, 1.0])
        self.ax[1].set_title('Average Precision=%0.2f' % sklearn.metrics.average_precision_score(self.y_true, self.y_score))


def run_simulations_and_make_plots():
    control_df = pd.read_csv('sim_setups.csv',header=0)
    decision_thresh = 0.5
    control_df['Sim_AUROCs']=''
    control_df['Sim_AvgPrecs']=''
    for idx in control_df.index.values.tolist():
        plotted = False
        sim_aurocs = []
        sim_avgprecs = []
        for i in range(10):
            y_true, y_score = add_true_positives([],[],n=control_df.at[idx,'TP'],decision_thresh=decision_thresh) #add 50 true positives
            y_true, y_score = add_false_positives(y_true,y_score,n=control_df.at[idx,'FP'],decision_thresh=decision_thresh)
            y_true, y_score = add_true_negatives(y_true,y_score,n=control_df.at[idx,'TN'],decision_thresh=decision_thresh)
            y_true, y_score = add_false_negatives(y_true,y_score,n=control_df.at[idx,'FN'],decision_thresh=decision_thresh) #remove 50 false positives
            sim_aurocs.append(round(sklearn.metrics.roc_auc_score(y_true,y_score),3))
            sim_avgprecs.append(round(sklearn.metrics.average_precision_score(y_true,y_score),3))
            if not plotted:
                title = control_df.at[idx,'Title']+'. When d='+str(decision_thresh)+': '+confusion_matrix_string(y_true, y_score,decision_thresh=decision_thresh)
                savetitle = control_df.at[idx,'SaveTitle']+'-d'+str(decision_thresh)
                PlotAll(title,savetitle,y_true,y_score)
                plotted = True
        control_df.at[idx,'Sim_AUROCs'] = str(sim_aurocs)
        control_df.at[idx,'Sim_AvgPrecs'] = str(sim_avgprecs)
    control_df.to_csv('sim_results.csv')

if __name__=='__main__':
    run_simulations_and_make_plots()
    