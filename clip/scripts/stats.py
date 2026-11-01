"""Statistics over entry_scores.tsv.

1  are experimental (crystallographic) interfaces on the RBP's known motif?
2  does AF3 place its interface on the same motif?
3  do AF3's own quality metrics predict motif agreement?
4  provenance: in-vitro motifs (assay-matched) vs CLIP motifs (in vivo)
5  gene clustering control
6  organism: human vs mouse (motifs are human; mouse tests ortholog transfer)
"""
import csv, math, collections, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
NUM = None

def fl(v):
    try: return float(v)
    except Exception: return float('nan')

rows = list(csv.DictReader(open(ROOT/'results/entry_scores.tsv'), delimiter='\t'))
CLS = {r['gene']: r['class'] for r in
       csv.DictReader(open(ROOT/'scripts/rbp_class.tsv'), delimiter='\t')}
for r in rows:
    r['class'] = CLS.get(r['gene'], '?')
NUM = [k for k in rows[0] if k.startswith(('score_','z_','p_','d_score','d_z','n_fp','maxfp','n_motif'))] + \
      ['af3_rna_ipTM','af3_rna_pTM','af3_chain_pair_pae_min','af3_ranking_score',
       'RNAmotif_score','RNAmatch_score','RNA_LDDT','iLDDT','RNA_TM','DockQ_Nr']
for r in rows:
    for k in NUM:
        r[k] = fl(r.get(k, ''))

def ok(r, arm, t):   return not math.isnan(r[f'z_{arm}_{t}'])
def both(rs, arm):   return [r for r in rs if ok(r, arm, 'exp') and ok(r, arm, 'pred')]

COH = {"post": "post-cutoff (novel to AF3)", "pre": "pre-cutoff (AF3 trained on)"}
print("="*78); print("0. COHORTS"); print("="*78)
for c in ("post", "pre"):
    sub = [r for r in rows if r["cohort"] == c]
    print(f"  {c:<5} {COH[c]:<30} {len(sub):>4} entries, "
          f"{len({r['gene'] for r in sub}):>3} genes, "
          f"{sum(1 for r in sub if ok(r,'all','exp')):>4} scored")
print()
print("="*78); print("1. IS THE EXPERIMENTAL (CRYSTAL) INTERFACE ON THE RBP'S KNOWN MOTIF?"); print("="*78)
e = [r for r in rows if ok(r, 'all', 'exp')]
ze = np.array([r['z_all_exp'] for r in e]); pe = np.array([r['p_all_exp'] for r in e])
print(f"  scored entries            : {len(e)} / {len(rows)}")
print(f"  mean z vs other-RBP null  : {ze.mean():+.2f}  (median {np.median(ze):+.2f})")
print(f"  one-sample t vs 0         : t={stats.ttest_1samp(ze,0).statistic:.2f}, p={stats.ttest_1samp(ze,0).pvalue:.2e}")
print(f"  Wilcoxon vs 0             : p={stats.wilcoxon(ze).pvalue:.2e}")
k = int((pe<0.05).sum())
print(f"  entries p<0.05            : {k}/{len(pe)} (chance {0.05*len(pe):.1f}); "
      f"binomial p={stats.binomtest(k,len(pe),0.05,alternative='greater').pvalue:.2e}")

print(); print("="*78); print("2. DOES AF3 PUT THE INTERFACE ON THE SAME MOTIF?"); print("="*78)
pr = both(rows,'all')
a = np.array([r['z_all_exp'] for r in pr]); b = np.array([r['z_all_pred'] for r in pr])
print(f"  paired entries            : {len(pr)}")
print(f"  mean z_exp {a.mean():+.2f} -> z_pred {b.mean():+.2f}  (delta {b.mean()-a.mean():+.2f})")
print(f"  paired t / Wilcoxon       : p={stats.ttest_rel(a,b).pvalue:.3f} / {stats.wilcoxon(a,b).pvalue:.3f}")
rp, rs_ = stats.pearsonr(a,b), stats.spearmanr(a,b)
print(f"  Pearson r / Spearman rho  : {rp.statistic:.3f} (p={rp.pvalue:.1e}) / {rs_.statistic:.3f} (p={rs_.pvalue:.1e})")
print("\n  --- AF3 fragments the footprint below the 4 nt needed to read a motif ---")
for r in [x for x in rows if ok(x,'all','exp') and not ok(x,'all','pred')]:
    print(f"    {r['gene']:<9}{r['pdb']:<6} exp z={r['z_all_exp']:+.2f} on '{r['fp_exp']}' vs {r['motif_exp']}"
          f"  ->  {int(r['n_fp_pred'])} AF3 footprints >=4 nt "
          f"(RNAmatch_score={r['RNAmatch_score']:.2f}, RNAmotif_score={r['RNAmotif_score']:.2f})")
print("  --- AF3 degrades the motif (dz < -0.4) ---")
for r in sorted([x for x in pr if x['d_z_all']<-0.4], key=lambda x:x['d_z_all']):
    print(f"    {r['gene']:<9}{r['pdb']:<6} z {r['z_all_exp']:+.2f} -> {r['z_all_pred']:+.2f} "
          f"({r['d_z_all']:+.2f})  '{r['fp_exp']}' -> '{r['fp_pred']}'")

print(); print("="*78)
print("3. DOES ANY QUALITY SIGNAL PREDICT MOTIF AGREEMENT?"); print("="*78)
print("  AF3 confidence = what the model reports about itself.")
print("  Computed metrics = accuracy/composition scored against the deposited structure.")
print("  DockQ_Nr is the mean RBP_DockQ over protein-RNA interfaces only; the")
print("  whole-complex Dockq is dominated by protein-protein interfaces.")
for title, mets in (("AF3 confidence",
                     ('af3_rna_ipTM','af3_rna_pTM','af3_chain_pair_pae_min','af3_ranking_score')),
                    ("computed metrics",
                     ('RNAmotif_score','RNAmatch_score','RNA_LDDT','iLDDT','RNA_TM','DockQ_Nr'))):
    print(f"\n  -- {title} --")
    print(f"  {'metric':<24}{'n':>4}{'r(metric,z_pred)':>19}{'p':>10}")
    for m in mets:
        v=[(r[m],r['z_all_pred']) for r in pr if not math.isnan(r[m])]
        if len(v)<4:
            print(f"  {m:<24}{len(v):>4}   (too few)"); continue
        c=stats.pearsonr([t[0] for t in v],[t[1] for t in v])
        print(f"  {m:<24}{len(v):>4}{c.statistic:>19.3f}{c.pvalue:>10.3f}")

print(); print("="*78)
print("4. PROVENANCE: IN-VITRO (assay-matched) vs CLIP (in vivo) MOTIFS")
print("="*78)
print("   PDB crystallography and an AF3 prediction of it are both in-vitro-like")
print("   observations of a binary complex; CLIP reports cellular binding.")
print(f"\n  {'arm':<9}{'n_exp':>6}{'z_exp':>8}{'n_pair':>7}{'z_pred':>8}{'d_z':>7}{'paired p':>10}{'r(e,p)':>8}")
for arm in ('invitro','clip','all'):
    ae=[r[f'z_{arm}_exp'] for r in rows if ok(r,arm,'exp')]
    pp=both(rows,arm)
    if not ae: continue
    if len(pp)>2:
        x=np.array([r[f'z_{arm}_exp'] for r in pp]); y=np.array([r[f'z_{arm}_pred'] for r in pp])
        pv=stats.wilcoxon(x,y).pvalue; rr=stats.pearsonr(x,y).statistic
        print(f"  {arm:<9}{len(ae):>6}{np.mean(ae):>8.2f}{len(pp):>7}{y.mean():>8.2f}"
              f"{y.mean()-x.mean():>7.2f}{pv:>10.3f}{rr:>8.3f}")
    else:
        print(f"  {arm:<9}{len(ae):>6}{np.mean(ae):>8.2f}{len(pp):>7}{'-':>8}{'-':>7}{'-':>10}{'-':>8}")
iv=[r['z_invitro_exp'] for r in rows if ok(r,'invitro','exp')]
cl=[r['z_clip_exp'] for r in rows if ok(r,'clip','exp')]
if len(cl)>2:
    u=stats.mannwhitneyu(iv,cl)
    print(f"\n  in-vitro vs CLIP arm, z_exp : Mann-Whitney U p={u.pvalue:.2e} "
          f"(in-vitro {np.mean(iv):+.2f} n={len(iv)}, CLIP {np.mean(cl):+.2f} n={len(cl)})")
    pairg=[r for r in rows if ok(r,'invitro','exp') and ok(r,'clip','exp')]
    if len(pairg)>2:
        x=np.array([r['z_invitro_exp'] for r in pairg]); y=np.array([r['z_clip_exp'] for r in pairg])
        print(f"  paired on the {len(pairg)} entries scored by BOTH arms: "
              f"in-vitro {x.mean():+.2f} vs CLIP {y.mean():+.2f}, Wilcoxon p={stats.wilcoxon(x,y).pvalue:.2e}")

print(); print("="*78); print("5. GENE-CLUSTERING CONTROL"); print("="*78)
gc = collections.Counter(r['gene'] for r in pr)
print(f"  {len(pr)} entries over {len(gc)} genes: " + ", ".join(f"{g}x{n}" for g,n in gc.most_common()))
print(f"\n  {'metric':<24}{'n_gene':>7}{'r_within':>10}{'p':>8}{'r_between':>11}{'p':>8}")
for m in ('af3_rna_ipTM','af3_rna_pTM','af3_chain_pair_pae_min','af3_ranking_score',
          'RNAmotif_score','RNAmatch_score','iLDDT','DockQ_Nr'):
    v=[(r['gene'],r[m],r['z_all_pred']) for r in pr if not math.isnan(r[m])]
    if len(v)<4: continue
    bg=collections.defaultdict(list)
    for g,x,y in v: bg[g].append((x,y))
    wx,wy=[],[]
    for g,pts in bg.items():
        if len(pts)<2: continue
        mx=np.mean([p[0] for p in pts]); my=np.mean([p[1] for p in pts])
        wx += [p[0]-mx for p in pts]; wy += [p[1]-my for p in pts]
    bx=[np.mean([p[0] for p in pts]) for pts in bg.values()]
    by_=[np.mean([p[1] for p in pts]) for pts in bg.values()]
    cw = stats.pearsonr(wx,wy) if len(wx)>3 else None
    cb = stats.pearsonr(bx,by_)
    sw = f"{cw.statistic:>10.3f}{cw.pvalue:>8.3f}" if cw else f"{'n/a':>10}{'':>8}"
    print(f"  {m:<24}{len(bg):>7}{sw}{cb.statistic:>11.3f}{cb.pvalue:>8.3f}")
print(f"\n  {'gene':<10}{'org':<7}{'n':>3}{'z_exp':>8}{'z_pred':>8}{'RNAmotif':>10}{'DockQ_Nr':>10}{'RNA_LDDT':>10}")
bg=collections.defaultdict(list)
for r in pr: bg[r['gene']].append(r)
def mn(rs,k):
    v=[r[k] for r in rs if not math.isnan(r[k])]; return np.mean(v) if v else float('nan')
for g,rs in sorted(bg.items(), key=lambda kv:-mn(kv[1],'z_all_exp')):
    _o = {r['organism'] for r in rs}
    _ol = 'mixed' if len(_o) > 1 else next(iter(_o))
    print(f"  {g:<10}{_ol:<7}{len(rs):>3}{mn(rs,'z_all_exp'):>8.2f}{mn(rs,'z_all_pred'):>8.2f}"
          f"{mn(rs,'RNAmotif_score'):>10.2f}{mn(rs,'DockQ_Nr'):>10.2f}{mn(rs,'RNA_LDDT'):>10.2f}")

print(); print("="*78)
print("6. RBP RECOGNITION CLASS  (the axis that actually separates)"); print("="*78)
print(f"  {'class':<11}{'n':>4}{'z_exp':>8}{'z_pred':>8}   genes")
for c in ('sequence','structure','guide'):
    v =[r['z_all_exp']  for r in rows if r['class']==c and ok(r,'all','exp')]
    vp=[r['z_all_pred'] for r in rows if r['class']==c and ok(r,'all','pred')]
    g = sorted({r['gene'] for r in rows if r['class']==c})
    if v:
        print(f"  {c:<11}{len(v):>4}{np.mean(v):>8.2f}"
              f"{(np.mean(vp) if vp else float('nan')):>8.2f}   {', '.join(g)}")
sq=[r['z_all_exp'] for r in rows if r['class']=='sequence' and ok(r,'all','exp')]
ns=[r['z_all_exp'] for r in rows if r['class'] in ('structure','guide') and ok(r,'all','exp')]
print(f"  sequence vs non-sequence  : Mann-Whitney p={stats.mannwhitneyu(sq,ns).pvalue:.2e}")

print(); print("="*78)
print("7. ORGANISM: HUMAN vs MOUSE (motifs are human; mouse = ortholog transfer)")
print("="*78)
for lab,t in (("exp (crystal)","exp"),("pred (AF3)","pred")):
    h=[r[f'z_all_{t}'] for r in rows if r['organism']=='human' and ok(r,'all',t)]
    m_=[r[f'z_all_{t}'] for r in rows if r['organism']=='mouse' and ok(r,'all',t)]
    if h and m_:
        print(f"  ALL entries      {lab:<15} human {np.mean(h):+.2f} (n={len(h)})  "
              f"mouse {np.mean(m_):+.2f} (n={len(m_)})  p={stats.mannwhitneyu(h,m_).pvalue:.4f}")
_m = [r for r in rows if r['organism']=='mouse' and ok(r,'all','exp')]
_ms = collections.Counter(r['gene'] for r in _m).most_common(1)
print(f"  -> confounded: {_ms[0][1]} of {len(_m)} mouse entries are {_ms[0][0]}, "
      f"a {CLS.get(_ms[0][0],'?')}-class RBP.")
for lab,t in (("exp (crystal)","exp"),("pred (AF3)","pred")):
    h=[r[f'z_all_{t}'] for r in rows if r['organism']=='human' and r['class']=='sequence' and ok(r,'all',t)]
    m_=[r[f'z_all_{t}'] for r in rows if r['organism']=='mouse' and r['class']=='sequence' and ok(r,'all',t)]
    if h and m_:
        print(f"  sequence-only    {lab:<15} human {np.mean(h):+.2f} (n={len(h)})  "
              f"mouse {np.mean(m_):+.2f} (n={len(m_)})  p={stats.mannwhitneyu(h,m_).pvalue:.4f}")
_sq = sorted({r['gene'] for r in rows
              if r['organism']=='mouse' and r['class']=='sequence' and ok(r,'all','exp')})
print(f"  -> only {', '.join(_sq)} remain: the effect does not survive, so no")
print("     ortholog-transfer penalty can be claimed.")
gm=collections.defaultdict(set)
for r in rows: gm[r['organism']].add(r['gene'])
for o in ('human','mouse'):
    print(f"  {o:<7} genes: {', '.join(sorted(gm[o]))}")

print(); print("="*78)
print("8. VIRAL ENTRIES"); print("="*78)
print("  exp_protein_rna is dominated by viral RNPs (101 SARS-CoV-2 chains, plus")
print("  influenza, Thogoto, Lassa, hMPV, TiLV). None of the four motif atlases")
print("  contains a viral RBP: ATtRACT spans 38 organisms, all cellular; RBPBench,")
print("  CISBP-RNA and RBPDB are host RBPs only. Scoring viral interfaces would")
print("  require deriving motifs de novo from viral CLIP (e.g. SARS-CoV-2 N), which")
print("  is outside the atlas-join design used here. Reported as uncovered, n=0.")


print(); print("="*78)
print("9. PRE- vs POST-CUTOFF COHORTS"); print("="*78)
print("  Does the CLIP/atlas motif signal hold on structures AF3 was trained on,")
print("  and does training exposure change how well AF3 lands on the motif?")
print()
print(f"  {'cohort':<6}{'n_exp':>6}{'z_exp':>8}{'n_pair':>7}{'z_pred':>8}{'d_z':>7}"
      f"{'paired p':>10}{'r(exp,pred)':>13}")
for c in ("post", "pre"):
    sub = [r for r in rows if r["cohort"] == c]
    ae = [r['z_all_exp'] for r in sub if ok(r,'all','exp')]
    pp = [r for r in sub if ok(r,'all','exp') and ok(r,'all','pred')]
    x = np.array([r['z_all_exp'] for r in pp]); y = np.array([r['z_all_pred'] for r in pp])
    print(f"  {c:<6}{len(ae):>6}{np.mean(ae):>8.2f}{len(pp):>7}{y.mean():>8.2f}"
          f"{y.mean()-x.mean():>7.2f}{stats.wilcoxon(x,y).pvalue:>10.3f}"
          f"{stats.pearsonr(x,y).statistic:>13.3f}")
ap = [r['z_all_exp'] for r in rows if r['cohort']=='post' and ok(r,'all','exp')]
bp = [r['z_all_exp'] for r in rows if r['cohort']=='pre'  and ok(r,'all','exp')]
print(f"\n  experimental interfaces on-motif in BOTH cohorts:")
for c, v in (("post", ap), ("pre", bp)):
    t = stats.ttest_1samp(v, 0)
    print(f"    {c:<5} mean z={np.mean(v):+.2f}  t={t.statistic:.2f}  P={t.pvalue:.2e}  n={len(v)}")
print(f"    cohort difference (z_exp): Mann-Whitney P={stats.mannwhitneyu(ap,bp).pvalue:.3f}")
dp = [r['d_z_all'] for r in rows if r['cohort']=='post' and not math.isnan(r['d_z_all'])]
db_ = [r['d_z_all'] for r in rows if r['cohort']=='pre'  and not math.isnan(r['d_z_all'])]
print(f"\n  AF3 motif shift (z_pred - z_exp):")
print(f"    post {np.mean(dp):+.3f} (n={len(dp)})   pre {np.mean(db_):+.3f} (n={len(db_)})"
      f"   Mann-Whitney P={stats.mannwhitneyu(dp,db_).pvalue:.3f}")
print("\n  by recognition class and cohort (z_exp / z_pred):")
print(f"  {'class':<11}{'cohort':<6}{'n':>4}{'z_exp':>8}{'z_pred':>8}")
for cl in ('sequence','structure','guide'):
    for c in ('post','pre'):
        v =[r['z_all_exp']  for r in rows if r['class']==cl and r['cohort']==c and ok(r,'all','exp')]
        vp=[r['z_all_pred'] for r in rows if r['class']==cl and r['cohort']==c and ok(r,'all','pred')]
        if v:
            print(f"  {cl:<11}{c:<6}{len(v):>4}{np.mean(v):>8.2f}"
                  f"{(np.mean(vp) if vp else float('nan')):>8.2f}")
