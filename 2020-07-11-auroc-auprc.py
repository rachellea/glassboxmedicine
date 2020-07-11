#2020-07-11-auroc-auprc.py

import numpy as np

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
        plt.savefig(savetitle+'.png')

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

def plot_equal():
    y_true, y_score = add_true_positives([],[],n=100,decision_thresh=0.5)
    y_true, y_score = add_false_positives(y_true,y_score,n=100,decision_thresh=0.5)
    y_true, y_score = add_true_negatives(y_true,y_score,n=100,decision_thresh=0.5)
    y_true, y_score = add_false_negatives(y_true,y_score,n=100,decision_thresh=0.5)
    title = 'Balanced Data. When d=0.5: '+confusion_matrix_string(y_true, y_score,decision_thresh=0.5)
    savetitle = 'Balanced'
    PlotAll(title,savetitle,y_true,y_score)

#######################
# High True Positives #---------------------------------------------------------
#######################
def plot_high_TP():
    for decision_thresh in [0.1,0.5,0.9]:
        plot_high_TP_constant_P(decision_thresh)
        plot_high_TP_high_P(decision_thresh)
        plot_high_TP_low_P(decision_thresh)
    
def plot_high_TP_constant_P(decision_thresh):
    y_true, y_score = add_true_positives([],[],n=150,decision_thresh=decision_thresh) #add 50 true positives
    y_true, y_score = add_false_positives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_true_negatives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_false_negatives(y_true,y_score,n=50,decision_thresh=decision_thresh) #remove 50 false positives
    title = 'High TP Constant P. When d='+str(decision_thresh)+': '+confusion_matrix_string(y_true, y_score,decision_thresh=decision_thresh)
    savetitle = 'HighTPConstantP'+str(decision_thresh)
    PlotAll(title,savetitle,y_true,y_score)

def plot_high_TP_high_P(decision_thresh):
    y_true, y_score = add_true_positives([],[],n=150,decision_thresh=decision_thresh) #add 50 true positives
    y_true, y_score = add_false_positives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_true_negatives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_false_negatives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    title = 'High TP High P. When d='+str(decision_thresh)+': '+confusion_matrix_string(y_true, y_score,decision_thresh=decision_thresh)
    savetitle = 'HighTPHighP'+str(decision_thresh)
    PlotAll(title,savetitle,y_true,y_score)
    
def plot_high_TP_low_P(decision_thresh):
    y_true, y_score = add_true_positives([],[],n=50,decision_thresh=decision_thresh) #From their high of 150, reduce by a factor of 3
    y_true, y_score = add_false_positives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_true_negatives(y_true,y_score,n=100,decision_thresh=decision_thresh)
    y_true, y_score = add_false_negatives(y_true,y_score,n=16,decision_thresh=decision_thresh) #3x fewer false positives than true positives
    title = 'High TP Low P. When d='+str(decision_thresh)+': '+confusion_matrix_string(y_true, y_score,decision_thresh=decision_thresh)
    savetitle = 'HighTPLowP'+str(decision_thresh)
    PlotAll(title,savetitle,y_true,y_score)
    
########################
# High False Positives #-------------------------------------------------------- 
########################






if __name__=='__main__':
    #plot_equal()
    plot_high_TP()
    