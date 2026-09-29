from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parent
# Reuse plotting style and data checks, replace the scientific panels below.
src=(ROOT/'_revision_260926_real_work/figures.py').read_text(encoding='utf-8')
src=src.replace("pd.read_csv(W/'metrics.csv')","pd.read_csv(ROOT/'_revision_260926_real_work/metrics.csv')").replace("pd.read_csv(W/'effects.csv')","pd.read_csv(ROOT/'_revision_260926_real_work/effects.csv')").replace("pd.read_csv(W/'clinical_sensitivity.csv')","pd.read_csv(ROOT/'_revision_260926_real_work/clinical_sensitivity.csv')")
src=src.replace("'o-'","'o'")
src=src.replace("'crDRP NNLS'","'crDRP-NNLS'").replace("'crDRP bulk'","'crDRP-bulk'").replace("'crDRP BayesPrism-fixed'","'crDRP-BayesPrism-fixed'")
src=src.replace("methods=['crdrp_bulk','crdrp_full_nnls','transformer_bulk','transformer_full_nnls','global_xgb','ridge','drug_mean']","methods=['crdrp_bulk','crdrp_full_nnls','global_xgb','ridge','drug_mean']")
src=src.replace("Same-count NNLS / BayesPrism-fixed / bulk","Same-count NNLS / BayesPrism / bulk").replace("Per-drug r + RMSE","OOF + within-fold per-drug r + RMSE")
exec(compile(src,str(Path(__file__)),'exec'),globals())
FM=pd.read_csv(W/'fold_metrics.csv');X=pd.read_csv(W/'xgb/metrics.csv');BM=pd.read_csv(ROOT/'260925_Re/runs/real_gdsc_cell_20260929_170425/metrics.csv')
# Rebuild primary plots with separated categorical points and both metrics.
fig,axs=plt.subplots(2,2,figsize=(11,8.4),layout='constrained')
for ax,key,title in [(axs[0,0],'cell','a  Cell holdout: concatenated OOF'),(axs[1,0],'pair','c  Pair holdout: concatenated OOF'),(axs[1,1],'drug','d  Drug holdout: concatenated OOF')]:profile(ax,key,methods,title)
ax=axs[0,1]
for j,(s,c) in enumerate(zip(seeds,C)):
 q=FM[(FM.analysis=='cell')&(FM.seed==s)].set_index('method').reindex(methods[:-1]);ax.scatter(np.arange(len(q))+(j-1)*.14,q.within_fold_median,c=c,s=25,label=f'Seed {s}')
ax.set_xticks(range(4),[LABEL[m] for m in methods[:-1]],rotation=32,ha='right');ax.axhline(0,c='#aaa',lw=.8);ax.set_ylabel('Median within-fold Fisher-z r');ax.set_title('b  Cell holdout: within-fold ranking',loc='left',weight='bold',fontsize=11)
axs[0,0].legend(frameon=False,ncol=3,fontsize=8);axs[1,1].text(.98,.02,'Drug mean: undefined',ha='right',transform=axs[1,1].transAxes,fontsize=8);save(fig,'Figure2_GDSC')
ME=pd.read_csv(W/'mixture_effects.csv');ME=ME[ME.metric.eq('median_paired_drug_delta_r')]
fig,axs=plt.subplots(1,3,figsize=(12,4.4),layout='constrained');profile(axs[0],'mixture',['crdrp_oracle','crdrp_bayesprism_fixed','crdrp_nnls','crdrp_bulk'],'a  Synthetic response recovery')
for ax,a,title in [(axs[1],'crdrp_oracle','b  Oracle − bulk'),(axs[2],'crdrp_bayesprism_fixed','c  BayesPrism-fixed − bulk')]:
 q=ME[ME.method_a.eq(a)].sort_values('seed')
 for j,(_,r) in enumerate(q.iterrows()):ax.errorbar(r.effect_a_minus_b,j,xerr=[[r.effect_a_minus_b-r.ci_low],[r.ci_high-r.effect_a_minus_b]],fmt='o',c=C[j],capsize=3)
 ax.axvline(0,c='#aaa',ls='--');ax.set_yticks(range(3),[f'Seed {s}' for s in seeds]);ax.set_xlabel('Median paired Δr (95% CI)');ax.set_title(title,loc='left',weight='bold',fontsize=11)
axs[0].legend(frameon=False,fontsize=8);save(fig,'Figure3_deconvolution')
fig,axs=plt.subplots(1,3,figsize=(12,4.5),layout='constrained');ms=['crdrp_full','crdrp_bulk','crdrp_full_shuffled'];ep=pd.read_csv(W/'nnls_epithelial_errors.csv')
for j,(s,c) in enumerate(zip(seeds,C)):
 q=pd.DataFrame(old['positive']);q=q[q.seed.eq(s)].set_index('method').reindex(ms);axs[0].scatter(np.arange(3)+(j-1)*.1,q.perdrug_pearson_median,c=c,s=26,label=f'Seed {s}')
 q=ep[ep.seed.eq(s)].set_index('compartment').reindex(['malignant','normal','combined_epithelial']);axs[1].scatter(np.arange(3)+(j-1)*.1,q.pearson,c=c,s=26)
 axs[2].scatter(np.arange(2)+(j-1)*.1,q.signed_bias.iloc[:2],c=c,s=26)
for ax,labels,title,ylabel in [(axs[0],['crDRP oracle','crDRP-bulk','Oracle shuffled'],'a  Known-signal control','Median per-drug r'),(axs[1],['Malignant','Normal','Combined epithelial'],'b  Held-out donor recovery','Fraction Pearson r'),(axs[2],['Malignant','Normal'],'c  Direction of NNLS error','Mean estimated minus true fraction')]:
 ax.set_xticks(range(len(labels)),labels,rotation=28,ha='right');ax.axhline(0,c='#aaa',lw=.8);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',weight='bold',fontsize=11)
axs[0].legend(frameon=False,fontsize=8);save(fig,'Figure4_positive_control')
# Full Transformer diagnostics remain visible, but outside the main evidence.
fig,axs=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
for ax,key,ms,title in [(axs[0],'positive',['transformer_full','transformer_bulk','transformer_full_shuffled'],'a  Known-signal control'),(axs[1],'diagnostic',['transformer_full_zeroZ','transformer_full_zeroZ_shuffled'],'b  Zero-Z diagnostic')]:
 for j,(s,c) in enumerate(zip(seeds,C)):
  q=pd.DataFrame(old[key]);q=q[q.seed.eq(s)].set_index('method').reindex(ms);ax.scatter(np.arange(len(ms))+(j-1)*.1,q.perdrug_pearson_median,c=c,s=25,label=f'Seed {s}')
 ax.set_xticks(range(len(ms)),['Full','Bulk','Full shuffled'] if key=='positive' else ['Full zero-Z','Zero-Z shuffled'],rotation=28,ha='right');ax.set_title(title,loc='left',weight='bold',fontsize=11);ax.axhline(0,c='#aaa',lw=.8);ax.set_ylabel('Median per-drug r')
for j,(s,c) in enumerate(zip(seeds,C)):
 for k,m in enumerate(['transformer_full_nnls','transformer_bulk']):
  q=M[(M.seed==s)&M.method.eq(m)].set_index('analysis').reindex(['cell','pair','drug']);axs[2].scatter(np.arange(3)+(j-1)*.12+(k-.5)*.035,q.median_r,c=c,marker='o' if k==0 else 'x',s=25)
axs[2].set_xticks(range(3),['Cell','Pair','Drug']);axs[2].set_title('c  Real GDSC: ○ full, × bulk',loc='left',weight='bold',fontsize=11);axs[2].set_ylabel('Median per-drug OOF r');axs[0].legend(frameon=False,fontsize=8);save(fig,'FigureS5_transformer')
fig,axs=plt.subplots(1,3,figsize=(12,4.5),layout='constrained');cm=pd.read_csv(ROOT/'260925_Re/runs/carcinoma_transfer_20260929_165047/metrics.csv')
for j,(s,c) in enumerate(zip(seeds,C)):
 q=X[X.seed.eq(s)].set_index('method');axs[0].scatter(np.arange(2)+(j-1)*.1,q.loc[['bulk_xgb_pca116','compartment_xgb'],'perdrug_pearson_median'],c=c,s=25,label=f'Seed {s}');axs[1].scatter(np.arange(4)+(j-1)*.1,q.loc[[f'bulk_xgb_pca{n}' for n in [32,64,116,256]],'perdrug_pearson_median'],c=c,s=25)
 q=M[(M.analysis=='transfer')&(M.seed==s)].set_index('method');v=cm[cm.seed.eq(s)].set_index('method');axs[2].scatter(np.arange(4)+(j-1)*.1,[q.loc['crdrp_bulk_transfer','median_r'],q.loc['crdrp_full_nnls_transfer','median_r'],v.loc['crdrp_bulk_transfer','perdrug_pearson_median'],v.loc['crdrp_full_nnls_transfer','perdrug_pearson_median']],c=c,s=25)
for ax,l,t in [(axs[0],['Bulk XGB','Compartment XGB'],'a  Tree representation'),(axs[1],['32','64','116','256 → 132'],'b  Drug PCA dimension'),(axs[2],['All bulk','All NNLS','Carcinoma bulk','Carcinoma NNLS'],'c  Source pretraining')]:
 ax.set_xticks(range(len(l)),l,rotation=28,ha='right');ax.set_title(t,loc='left',weight='bold',fontsize=11);ax.set_ylabel('Median per-drug OOF r');ax.axhline(0,c='#aaa',lw=.8)
axs[0].legend(frameon=False,fontsize=8);save(fig,'FigureS6_sensitivity')
fig,axs=plt.subplots(1,3,figsize=(12,4.1),layout='constrained')
for ax,file,ycol,title in [(axs[0],'ABSOLUTE_matched.csv','purity','a  ABSOLUTE (n = 998)'),(axs[1],'CPE_matched.csv','CPE','b  CPE (n = 1,062)')]:
 q=pd.read_csv(W/'clinical'/file);ax.scatter(q.malignant_epithelial,q[ycol],s=7,alpha=.25,c=C[0],edgecolors='none');ax.set(xlim=(0,1),ylim=(0,1),xlabel='NNLS malignant RNA contribution',ylabel=ycol);ax.set_title(title,loc='left',weight='bold',fontsize=11)
q=pd.read_csv(W/'clinical/PFI_cox.csv');axs[2].errorbar(q.HR,range(8),xerr=[q.HR-q.CI_low,q.CI_high-q.HR],fmt='o',c=C[2],capsize=3);axs[2].set_yticks(range(8),q.compartment.str.replace('_',' '));axs[2].invert_yaxis();axs[2].axvline(1,c='#aaa',ls='--');axs[2].set_xlabel('PFI HR per SD (95% CI)');axs[2].set_title('c  Age/stage-adjusted PFI',loc='left',weight='bold',fontsize=11);save(fig,'FigureS7_purity')
fig,ax=plt.subplots(figsize=(7.5,4.3),layout='constrained')
for j,(s,c) in enumerate(zip(seeds,C)):
 q=M[(M.analysis=='count')&(M.seed==s)].set_index('method');r=BM[BM.seed.eq(s)].iloc[0];v=[q.loc['crdrp_bulk','median_r'],q.loc['crdrp_full_nnls','median_r'],q.loc['crdrp_full_bayesprism_fixed','median_r'],r.perdrug_pearson_median];ax.scatter(np.arange(4)+(j-1)*.13,v,c=c,s=28,label=f'Seed {s}')
ax.set_xticks(range(4),['crDRP-bulk','crDRP-NNLS','crDRP-BayesPrism-fixed','crDRP-BayesPrism-updated'],rotation=20,ha='right');ax.set_ylabel('Median per-drug OOF r');ax.axhline(0,c='#aaa',lw=.8);ax.legend(frameon=False);ax.set_title('Count-matched real GDSC cell holdout',loc='left',weight='bold');save(fig,'FigureS8_bayesprism')
