"""Matched mCH mechanism comparison from saved precommit diagnostics and actions."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .transfer_publication import _style

DIAGNOSTICS = [('temporal', '#1769AA', r'$\eta_t/B_n$'),
               ('representation', '#D55E00', r'$\eta_r/B_n$'),
               ('invariant', '#7B2CBF', r'$\eta_I/I_{\max}$'),
               ('sharpness', '#A88416', r'$s_{\rm loc}$')]


def validate_log(log, data):
    amplitudes = []
    for i, case in enumerate(log['cases']):
        time = data[f'time_{i}']; rows = case['rows']
        assert len(time) == len(rows)+1 == len(case['particles'])
        assert np.isclose(time[0], 0) and np.isclose(time[-1], log['T'])
        assert np.all(np.diff(time)>0)
        assert np.allclose(np.diff(time),[r['dt'] for r in rows],rtol=1e-10,atol=1e-14)
        amplitudes.append(float(data[f'field_{i}'][0].max()))
        previous_N = log['initial_N']
        for j,r in enumerate(rows):
            assert np.isclose(r['start'],time[j]) and np.isclose(r['time'],time[j+1])
            assert r['action'] in ('hold','shrink_time','refine_space')
            if r['action']=='shrink_time': assert 0<r['dt']<log['base_dt']
            else: assert np.isclose(r['dt'],log['base_dt'])
            assert r['N'] == previous_N*(2 if r['action']=='refine_space' else 1)
            previous_N=r['N']
            assert any(c['action']==r['action'] and np.isclose(c['dt'],r['dt']) for c in r['candidates'])
            d=r['diagnostics']
            assert d['migration'] is None
            assert d['budget']>0 and d['invariant_limit']>0
            assert all(np.isfinite(d[k]) and d[k]>=0 for k,_,_ in DIAGNOSTICS)
    assert np.isclose(amplitudes[0],amplitudes[1],rtol=1e-10)
    assert all(not p['q'] for p in log['cases'][0]['particles'])
    assert all(p['q'] for p in log['cases'][1]['particles'])


def plot_saved_mechanism(output):
    output=Path(output)
    log=json.loads((output/'decisions.json').read_text())
    data=np.load(output/'trajectories.npz',allow_pickle=False)
    validate_log(log,data)
    vals=[]
    for case in log['cases']:
        values=[]
        for r in case['rows']:
            d=r['diagnostics']
            values.append([d[k]/(d['invariant_limit'] if k=='invariant' else 1 if k=='sharpness' else d['budget']) for k,_,_ in DIAGNOSTICS])
        vals.append(np.array(values))
    maximum=max(1.,max(v.max() for v in vals))*2
    with mpl.rc_context(_style()):
        fig=plt.figure(figsize=(8.3,5.2))
        gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.35,1.1],wspace=.43,hspace=.38)
        fig.subplots_adjust(left=.10,right=.985,top=.92,bottom=.12)
        for i,case in enumerate(log['cases']):
            x=data['x']; field=data[f'field_{i}']; time=data[f'time_{i}']; rows=case['rows']
            tau=np.array([r['start'] for r in rows])/log['T']
            # Matched interior time, not error-selected or independently ranked.
            mid=int(np.argmin(abs(time/log['T']-.5)))
            physical=gs[i,0].subgridspec(2,1,height_ratios=[2.5,1],hspace=.18)
            ax=fig.add_subplot(physical[0]); strip=fig.add_subplot(physical[1])
            for j,col,ls in [(0,'#8D99AE',':'),(mid,'#1769AA','--'),(len(time)-1,'#20242A','-')]:
                ax.plot(x,field[j],color=col,ls=ls,lw=1.1,label=rf'$\tau={time[j]/log["T"]:.2g}$')
            ax.set(ylabel='$u_h$',ylim=(.1,1.03)); ax.tick_params(labelbottom=False)
            ax.legend(loc='upper right',fontsize=6,frameon=False)
            strip.plot(x,data[f'momentum_{i}'][mid],color='#1769AA',lw=.8)
            strip.set(xlabel='$x$',ylabel='$m_r$')
            p=case['particles'][mid]
            if p['q']:
                # Marker locations are saved q_j; no delta height is invented.
                strip.plot(p['q'],[.88]*len(p['q']),'^',color='#009E73',ms=4,
                           transform=strip.get_xaxis_transform())
                strip.text(.98,.83,'▲ particle locations',transform=strip.transAxes,ha='right',fontsize=5.7,color='#009E73')
            diag=fig.add_subplot(gs[i,1])
            for j,(key,color,label) in enumerate(DIAGNOSTICS):
                # Exact zero is masked on log axes, not turned into a fake positive indicator.
                y=np.ma.masked_less_equal(vals[i][:,j],0)
                diag.plot(tau,y,color=color,lw=1,label=label)
            diag.set(yscale='log',ylim=(1e-12,maximum),xlim=(0,1),xlabel=r'$\tau=t/T$',ylabel='Budget ratios / sharpness')
            diag.axhline(1,color='#8D99AE',lw=.5,ls=':')
            diag.legend(loc='lower left',fontsize=6,frameon=False,ncol=2)
            if np.all(vals[i][:,1]==0):
                diag.text(.98,.48,r'$\eta_r=0$ (measured)',transform=diag.transAxes,ha='right',fontsize=6,color='#D55E00')
            tracks=gs[i,2].subgridspec(2,1,height_ratios=[2,1],hspace=.22)
            dtax=fig.add_subplot(tracks[0]); ledger=fig.add_subplot(tracks[1])
            dt=np.array([r['dt'] for r in rows])/log['base_dt']
            dtax.stairs(dt,np.r_[tau,1.],baseline=None,color='#20242A',lw=1.2)
            dtax.set(ylabel=r'$\Delta t_n/\Delta t_0$',ylim=(0,1.12),xlim=(0,1));dtax.tick_params(labelbottom=False)
            labels=np.array([r['action'] for r in rows]); changes=np.r_[0,np.flatnonzero((labels[1:]!=labels[:-1])|(~np.isclose(dt[1:],dt[:-1])))+1]
            palette={'hold':'#8D99AE','shrink_time':'#7B2CBF','refine_space':'#009E73'}
            for j in changes:
                color=palette[labels[j]]
                dtax.scatter(tau[j],dt[j],color=color,s=18,zorder=4)
                if labels[j]!='hold':diag.axvline(tau[j],color=color,lw=.6,ls='--')
            # Run-length ledger represents every commitment without repeating large markers.
            for left,right in zip(changes,np.r_[changes[1:],len(rows)]):
                a=labels[left]; end=tau[right] if right<len(rows) else 1.
                ledger.plot([tau[left],end],[0,0],color=palette[a],lw=2)
                ledger.scatter(tau[left],0,marker='o' if a=='hold' else 'D',s=20,color=palette[a],zorder=4)
                ledger.text((tau[left]+end)/2,.25,f'{a.replace("_"," ")} × {right-left}',ha='center',fontsize=6)
            fallback=np.flatnonzero([r['abstained'] for r in rows])
            if len(fallback):ledger.plot(tau[fallback],np.full(len(fallback),-.35),'o',mfc='none',mec='#8D99AE',ms=2)
            if len(fallback):ledger.text(.98,-.61,'○ model abstention',ha='right',fontsize=5.7,color='#59636E')
            ledger.set(xlim=(0,1),ylim=(-.65,.65),yticks=[],xlabel=r'$\tau=t/T$')
            levels=sorted(set([log['initial_N']]+[r['N'] for r in rows]))
            dtax.text(.5,.85,f'N = {levels[0]} throughout' if len(levels)==1 else 'N: '+' → '.join(map(str,levels)),transform=dtax.transAxes,ha='center',fontsize=6.5)
            for j,a in enumerate([ax,diag,dtax]):a.text(-.16,1.06,f'({chr(97+i*3+j)})',transform=a.transAxes,fontweight='bold',fontsize=9)
            if i==0:
                for a,title in zip([ax,diag,dtax],['Physical / representation state','Causal diagnostic signature','Committed numerical response']):a.set_title(title,fontsize=8)
            fig.text(.025,.73 if i==0 else .29,'Smooth regime' if i==0 else 'Peakon-containing regime',rotation=90,va='center',fontsize=8)
        for ext in ('png','pdf'):fig.savefig(output/f'mch_mechanism_comparison.{ext}',dpi=600,bbox_inches='tight',pad_inches=.03)
        plt.close(fig)
    (output/'caption.txt').write_text(
        'Matched mCH mechanism comparison. Model source: '+log['model_root']+'. This is not independent paper validation. '
        'Peak field amplitude (not peak-to-trough amplitude), domain, alpha, tolerance, initial N and physical horizon are matched. '
        'Profiles use native initial, nearest mid-horizon and final states. Triangles mark saved particle locations; the lower strip shows regular momentum, not delta functions. '
        'Diagnostics are selected-candidate trial estimates available before commitment, not direct measurements of neural attention. '
        'Temporal and interface estimates are divided by the same local budget B_n; invariant drift by its analytical limit; sharpness is already dimensionless. '
        'The unit line is a budget reference for ratios, not a threshold for sharpness. Migration is unavailable and omitted, not set to zero. '
        'Both rows share the diagnostic scale; exact zeros are omitted on the logarithmic axis. '
        'The action ledger is run-length encoded; every commitment is preserved in decisions.json. '
        'Open circles, if present, denote frozen-model abstention separately from the executed solver action. '
        'Shrink means selection relative to the registered base step, not recursive halving. No reference enters this experiment.\n',encoding='utf-8')
    (output/'integrity.json').write_text(json.dumps({'status':'PASS','scope':'logging and plotting integrity only; not accuracy certification','model_manifest':log['manifest_sha256']},indent=2))
