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

def confusion_matrix_values(y_true, y_score, decision_thresh):
    #Obtain binary predicted labels by applying <decision_thresh> to <y_score>
    y_pred = (np.array(y_score) > decision_thresh)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    return true_neg, false_pos, false_neg, true_pos

def confusion_matrix_string(y_true, y_score, decision_thresh):
    """Return the confusion matrix as a string"""
    true_neg, false_pos, false_neg, true_pos =  confusion_matrix_values(y_true, y_score, decision_thresh)
    return 'at d=0.5, tp='+str(true_pos)+', fn='+str(false_neg)+', tn='+str(true_neg)+', fp='+str(false_pos)

def calculate_tpr_fpr_prec(y_true, y_score, decision_thresh):
    true_neg, false_pos, false_neg, true_pos =  confusion_matrix_values(y_true, y_score, decision_thresh)
    tpr_recall = float(true_pos)/(true_pos + false_neg)
    fpr = float(false_pos)/(false_pos+true_neg)
    precision = float(true_pos)/(true_pos + false_pos)
    return tpr_recall, fpr, precision

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
        fig, self.ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8,5.5))
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        #Matplotlib tight layout doesn't take into account title so we pass
        #the rect argument
        #https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0,0.10,1,0.90])
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
        
        #Plot dots at certain decision thresholds, for clarity
        for d in [0.1, 0.5, 0.9]:
            tpr_recall, fpr, _ = calculate_tpr_fpr_prec(self.y_true, self.y_score, d)
            self.ax[0].plot(fpr, tpr_recall, 'o', color = self.roc_color)
            self.ax[0].annotate('d='+str(d), (fpr, tpr_recall))
    
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
        
        #Plot dots at certain decision thresholds, for clarity
        for d in [0.1, 0.5, 0.9]:
            tpr_recall, _, precision = calculate_tpr_fpr_prec(self.y_true, self.y_score, d)
            self.ax[1].plot(tpr_recall, precision, 'o', color = self.pr_color)
            text = self.ax[1].annotate('d='+str(d), (tpr_recall, precision))
            text.set_rotation(45)

def make_demo_initial_plots():
    y_true = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]*2
    y_score = [0.1,0.25,0.36,0.33,0.32,0.47,0.45,0.26,0.85,0.94,0.63,0.72,0.81,0.9,0.66,0.32,0.43,0.58,0.82,0.99]*2
    for d in [0.49,0.52,0.67]:
        y_true, y_score = add_true_positives(y_true,y_score,n=50,decision_thresh=d)
        y_true, y_score = add_true_negatives(y_true,y_score,n=50,decision_thresh=d)
    for idx in np.random.randint(low=0,high=len(y_true)-1,size=40):
        y_score[idx] = np.random.rand(1).tolist()[0]
    PlotAll('Example','Example',y_true,y_score)

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
            y_true, y_score = add_true_positives([],[],n=control_df.at[idx,'TP'],decision_thresh=decision_thresh)
            y_true, y_score = add_false_positives(y_true,y_score,n=control_df.at[idx,'FP'],decision_thresh=decision_thresh)
            y_true, y_score = add_true_negatives(y_true,y_score,n=control_df.at[idx,'TN'],decision_thresh=decision_thresh)
            y_true, y_score = add_false_negatives(y_true,y_score,n=control_df.at[idx,'FN'],decision_thresh=decision_thresh)
            sim_aurocs.append(round(sklearn.metrics.roc_auc_score(y_true,y_score),3))
            sim_avgprecs.append(round(sklearn.metrics.average_precision_score(y_true,y_score),3))
            if not plotted:
                title = 'Data'+control_df.at[idx,'DatasetProperty']+' / Model'+control_df.at[idx,'ClassifierProperty']
                savetitle = title.replace(' ','-').replace('/','-').replace('---','-')
                title = title+'\n'+confusion_matrix_string(y_true, y_score,decision_thresh=decision_thresh)
                PlotAll(title,savetitle,y_true,y_score)
                plotted = True
        control_df.at[idx,'Sim_AUROCs'] = str(sim_aurocs)
        control_df.at[idx,'Sim_AvgPrecs'] = str(sim_avgprecs)
    control_df.to_csv('sim_results.csv')

if __name__=='__main__':
    make_demo_initial_plots()
    run_simulations_and_make_plots()
    