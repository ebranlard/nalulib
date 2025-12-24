import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy import signal


class SignalTooShortException(Exception): pass

# --------------------------------------------------------------------------------}
# --- Transient detection
# --------------------------------------------------------------------------------{
def transientEnd_stats(t, x, W=None, W_frac=0.05, eps_mu_rel=0.0005, eps_sigma_rel=0.0005, eps_E_rel=0.0005, n_hold=5, plot=False):
    """ Detect transient time using moving window and change of statistics"""
    t = np.asarray(t)
    x = np.asarray(x)
    N = len(x)
    # --- Automatic window length
    if W is None:
        W = int(max(10, N*W_frac)) # Window length
    if W> len(x):
        raise SignalTooShortException('Signal too short')

    # --- Sliding statistics over window
    # Option 1 (expensive)
    #mu  = np.array([np.mean(x[i:i+W]) for i in range(N-W)])
    #sig = np.array([np.std (x[i:i+W]) for i in range(N-W)])
    #E   = np.array([ np.sqrt(np.mean(x[i:i+W]**2)) for i in range(N-W) ])
    # Option 2 (O(N))
    x2 = x*x
    cs  = np.cumsum(x)
    cs2 = np.cumsum(x2)
    mu  = (cs[W:] - cs[:-W]) / W
    var = (cs2[W:] - cs2[:-W]) / W - mu**2
    sig = np.sqrt(np.maximum(var, 0.0))
    E   = np.sqrt((cs2[W:] - cs2[:-W]) / W)


    # --- Automatic tolerances
    eps_mu    = eps_mu_rel   * np.std(x)
    eps_sigma = eps_sigma_rel * np.std(x)
    eps_E     = eps_E_rel     * np.mean(E)

    eps_mu_abs = 0.01 * np.std(mu)  

    #mu_ref = np.mean(mu)  # long-term target
    #mu_min = np.min(mu)
    #mu_max = np.max(mu)
    #mu_ref = mu[-1]  # long-term target
    #dmu_abs = np.abs(mu - mu[-1])
    #dmu_rel = dmu_abs / (np.std(mu) + 1e-12)
    #stable_mu = (dmu_abs < eps_mu_abs) | (dmu_rel < eps_mu_rel)
    dmu = np.abs(np.diff(mu))
    dsig = np.abs(np.diff(sig)) 
    dE   = np.abs(np.diff(E)) # dE/dt


    stable_mu  = dmu < eps_mu
    stable_sig = dsig < eps_sigma
    stable_E   = dE<eps_E
    stable = stable_mu & stable_sig & stable_E

    t_tr =None
    # Find indices where stable is False
    false_idx = np.flatnonzero(~stable)
    if len(false_idx)==0:
        print('[WARN] Steady-state criteria are always satisfied')
        t_tr=t[W+0]
    else:
        # The first index after the last False is the start of 'True forever'
        if (false_idx[-1] + 1) < len(stable):
            ii = false_idx[-1] + 1
            t_tr=t[W+ii]

    if plot:
        # --- Time axes (aligned with window end)
        t_stat = t[W:N]
        t_diff = t_stat[1:]

        fig, axes = plt.subplots( 8, 1, sharex=True, figsize=(6.5, 8))
        fig.subplots_adjust( left=0.12, right=0.95, top=0.97, bottom=0.05, hspace=0.10)
        # --- 1) Raw signal
        axes[0].plot(t, x, 'k', lw=1)
        axes[0].set_ylabel('x')
        axes[0].set_title('Transient detection diagnostics')
        # --- 2) Mean
        axes[1].plot(t_stat, mu, 'b')
        axes[1].set_ylabel(r'$\mu$')
        # --- 3) Std
        axes[2].plot(t_stat, sig, 'g')
        axes[2].set_ylabel(r'$\sigma$')
        # --- 4) Energy (RMS)
        axes[3].plot(t_stat, E, 'm')
        axes[3].set_ylabel('RMS')
        # --- 5) |d(mu)|
        axes[4].plot(t_diff, dmu, 'b')
        axes[4].axhline(eps_mu, color='b', ls='--', lw=1)
        axes[4].fill_between( t_diff, 0, dmu.max(), where=stable_mu, color='b', alpha=0.15)
        axes[4].set_ylabel(r'$|\Delta\mu|$')
        # --- 6) |d(sigma)|
        axes[5].plot(t_diff, dsig, 'g')
        axes[5].axhline(eps_sigma, color='g', ls='--', lw=1)
        axes[5].fill_between( t_diff, 0, dsig.max(), where=stable_sig, color='g', alpha=0.15)
        axes[5].set_ylabel(r'$|\Delta\sigma|$')
        # --- 7) |d(E)|
        axes[6].plot(t_diff, dE, 'm')
        axes[6].axhline(eps_E, color='m', ls='--', lw=1)
        axes[6].fill_between( t_diff, 0, dE.max(), where=stable_E, color='m', alpha=0.15)
        axes[6].set_ylabel(r'$|\Delta E|$')
        # --- Global stability (all criteria)
        axes[7].plot( t_diff, stable)
        #axes[7].fill_between( t_diff, 0, dE.max(), where=stable, color='lime', alpha=0.25, label='All criteria met')
        axes[7].set_ylabel(r'Criteria')
        axes[7].set_xlabel('Time')
        # --- Mark detected transient end
        if t_tr is not None:
            for ax in axes:
                ax.axvline(t_tr, color='r', lw=1.5, ls='--')
            axes[0].text( t_tr, axes[0].get_ylim()[1], ' transient end', color='r', va='top')

    return t_tr


def analyze_steady_state(
    t, x,
    tail_frac=0.5,
    head_frac_min=0.20,
    head_frac_max=0.4,
    rtol_const=1e-4,
    p_reached=0.9999,
    rtol_period=1e-3,
    min_periods=3,
    diverge_tol=5.0,
    plot=False
):
    """
    Steady-state analysis for scalar signals.
    Returns a dict describing:
      - diverging
      - steady constant
      - steady periodic
      - not converged
    """
    d_out={"type": "NA", "converged": False, "t_trans1":np.nan, "t_trans2": np.nan, "x_ss": np.nan, "period": np.nan, "n_periods": np.nan, "confidence": None}

    t = np.asarray(t)
    x = np.asarray(x)
    N = len(x)
    if N<2:
        print('[FAIL] steady_state.py: Less than 2 values, no determination of SS done.')
        return d_out
    elif N<30:
        print('[WARN] steady_state.py: Less than 30 values, determination of SS might be wrong.')
        plot=True
    dt = t[1] - t[0]

    # --- Detrend (linear)
    #x_dt = detrend(x, type="linear")

    # ---Divergence check (energy growth)
    n_tail = int(tail_frac * N)
    rms_head = np.sqrt(np.mean(x[:n_tail]**2))
    rms_tail = np.sqrt(np.mean(x[-n_tail:]**2))
    if rms_tail > diverge_tol * max(rms_head, 1e-12):
        d_out['type'] = 'diverging'
        return d_out

    # --------------------------------------------------------------------------------}
    # --- Estimate transient time 
    # --------------------------------------------------------------------------------{
    T_tot = t[-1] - t[0]
    t_min = head_frac_min*T_tot + t[0]
    try:
        t_trans = transientEnd_stats(t, x, plot=plot, eps_mu_rel=0.0001, eps_sigma_rel=0.001, eps_E_rel=0.001) 
        head_frac = head_frac_max
        if t_trans is not None:
            t_frac = (t_trans-t[0])/ T_tot
            if (t_frac<head_frac_min) or (t_frac>head_frac_max) :
                pass
                #print(f'[WARN] Transient time {t_frac*100:.1f}% not within {head_frac_min*100:.1f}%-{head_frac_max*100:.1f}%')
            else:
                head_frac = t_frac
        t_trans1 = head_frac*T_tot + t[0]
        b = t> t_trans1
        t = t[b]
        x = x[b]
        head_frac=0
        d_out['t_trans1'] = t_trans1
    except SignalTooShortException as e:
        print('[FAIL] steady_state.py: Cannot find stead-state, signal is too short')
        head_frac=0
        d_out['t_trans1'] = 0
    #print('t_trans', t_trans)
    #if t_trans is None:
    #    t_trans = transientEnd_stats(t, x, plot=plot, eps_mu_rel=0.01, eps_sigma_rel=0.01, eps_E_rel=0.01) 
    #    print('t_trans', t_trans)
    
    # --------------------------------------------------------------------------------}
    # --- Convergence to constant
    # --------------------------------------------------------------------------------{
    x_tail = x[-n_tail:]
    x_ss = np.mean(x_tail)
    sigma_tail = np.std(x_tail)

    scale = max(abs(x_ss), sigma_tail, 1e-12)
    r = np.abs(x - x_ss) / scale

    converges_to_constant = np.all(r[-n_tail:] < rtol_const)

    if converges_to_constant:
        tol_reached = 1.0 - p_reached
        idx = None
        for i in range(N):
            if np.all(r[i:] < tol_reached):
                idx = i
                break
        d_out['type'] = 'constant'
        d_out['converged']= True
        d_out['t_trans2']= t[idx] if idx is not None else None
        d_out['x_ss'] = x_ss
        return d_out

    # ------------------------------------------------------------
    # 3. Period estimation via FFT (full detrended signal)
    # ------------------------------------------------------------
    f0s, doms = detect_period_by_fft_dom(t, x, n=3, dom_thres=10, head_frac=head_frac, plot=plot)
    #print('T0', f0s, 1/np.array(f0s), doms)
    if len(f0s)>2:
        if np.around(f0s[0]/f0s[1], 1)==2 and doms[1]/doms[0]>0.2: 
            #print('>>> Swap')
            f0s = [f0s[1]]
            doms = [doms[1]]
    if len(f0s)==3:
        if np.all(np.around(np.sort(f0s/np.min(f0s)),2) == [1, 2, 3]):
            #print('>>> Three swap')
            ii = np.argmin(f0s)
            f0s =[f0s[ii]]
            doms=[doms[ii]]

    if len(f0s)==0:
        print('[WARN] steady_state.py: No dominant frequencies found.')
        f0 = np.nan
        dom = np.nan
    else:
        f0 = f0s[0]
        dom = doms[0]

    f1 = detect_period_by_peaks(t, x, T_guess=1/f0, plot=plot, head_frac=head_frac)
    T1=1/f1
    T0=1/f0
    if np.isnan(f0):
        period = 1.0 / f0
    else:
        period = 1.0 / f1 # Might be NaN
    #if abs(T1 - T0) / T0 > 0.3: # Check if difference > 30% of T0
    #    print('T0', 1/f0)
    #    print('T1', 1/f1)
    #    print("WARNING: The difference between T1 and T0 is larger than >30%")
    d_out['confidence'] = 'low'



    # No dominant frequency according to FFT
    if len(f0s)<= 0:
        d_out['type'] = 'not_converged_1'
        return d_out

    if np.isnan(period):
        d_out['type'] = 'not_converged_2'
        return d_out
# 
    # ------------------------------------------------------------
    # 4. Time-domain periodic validation (backward)
    # ------------------------------------------------------------
#     is_periodic, t_onset, n_periods = detect_periodic_by_correlation(t, x, period, rtol_corr=0.99, min_periods=3, plot=plot)
#     print('is_periodic 1: ', period, is_periodic, t_onset, n_periods, plot)
    is_periodic, t_onset, n_periods = detect_periodic_by_windows_overlap(t, x, period, rtol=0.10, min_periods=3, plot=plot, head_frac=head_frac)
    if not is_periodic:
        d_out['type'] = 'not_converged_3'
        return d_out
    # Periodic ss
    d_out['type']="steady_periodic"
    d_out['converged']= True
    d_out['t_trans2']=t_onset
    d_out['period']=period
    d_out['n_periods']=n_periods
    d_out['dominance']=dom
    return d_out


def detect_period_by_peaks(t, x, T_guess=None, plot=False, head_frac=0.1, head_frac_mean=0.5):
    from welib.tools.damping import peak_indexes
    dt = t[1]-t[0]
    min_dist=1
    if (T_guess is not None):
        if not np.isnan(T_guess):
            min_dist = max(1, int((T_guess/10)/dt) )
        #print('min_dist', min_dist)


    N=len(x)
    n_head = int(head_frac * N)
    xf = x[n_head:]
    tf = t[n_head:]

    x_mean = np.mean(x[int(head_frac_mean*N):])
    x_min  = np.min( x[int(head_frac_mean*N):])
    x_max  = np.max( x[int(head_frac_mean*N):])
    x_loc = (xf-x_min)/(x_max-x_min)-0.5 #-x_mean # Watch out copy it
    #threshold = np.mean(abs(x-x_mean))/3;
    #I =peak_indexes(x, thres=threshold, min_dist=min_dist, thres_abs=True)
    I1 =peak_indexes( x_loc, thres=0.9*0.5, min_dist=min_dist, thres_abs=True)
    I2 =peak_indexes(-x_loc, thres=0.9*0.5, min_dist=min_dist, thres_abs=True)
    # Estimating "index" period
    T1 = np.nan
    T2 = np.nan
    if len(I1)>1:
        iT1= round(np.median(np.diff(I1)));
        T1 = iT1*dt
    if len(I2)>1:
        iT2= round(np.median(np.diff(I2)));
        T2 = iT2*dt
    T = np.nanmean([T1,T2])
    f = 1/T

    if plot:
        fig,ax = plt.subplots(1, 1, sharey=False, figsize=(6.4,4.8))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
#         ax.plot(t, x, c=(0.5, 0.5, 0.5))
#         ax.plot(tf[I1], xf[I1], 'ko')
#         ax.plot(tf[I2], xf[I2], 'kd')

        ax.plot(tf, x_loc, c=(0.5, 0.5, 0.5))
        ax.plot(tf[I1], x_loc[I1], 'ko')
        ax.plot(tf[I2], x_loc[I2], 'kd')

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend()

    return f






def detect_period_by_fft_dom(t, x, n=3, dom_thres=10, head_frac=0.2, plot=False):
    dt = t[1] - t[0]
    N = len(x)
    n_head = int(head_frac * N)

    # discard head
    xf = x[n_head:]
    x_fft = xf - np.mean(xf)
    N = len(x_fft)
    freqs = np.fft.rfftfreq(N, dt)
    spectrum = np.abs(np.fft.rfft(x_fft))**2
    spectrum[0] = 0.0
    S_mean = np.mean(spectrum)+1e-12
    S_rel = spectrum/S_mean

    delta_f = freqs[1]-freqs[0]

    sorted_indices = np.argsort(S_rel)[::-1]

    # --- Method 1 the top n ones
    #top_I = sorted_indices[:n]
    #top_freqs = freqs[top_I]
    #top_dominances = S_rel[top_I]

    # --- Method 2, top freqs cannot be 2 delta f apart
    # Initialize lists to store the final, filtered top frequencies and dominances
    top_I = []
    top_freqs = []
    top_dominances = []
    min_distance_df = 2 # 2 * delta_f
    min_distance_hz = min_distance_df * delta_f
    # Iterate through ALL sorted peaks until we have exactly 'n' reliable ones
    for i_sort in sorted_indices:
        current_freq = freqs[i_sort]
        current_dominance = S_rel[i_sort]
        current_index = i_sort # The index within the original 'freqs' array

        if current_dominance == 0:
            continue # Skip zero-dominance bins
            
        is_too_close = False
        # Check against peaks ALREADY selected
        for j, selected_freq in enumerate(top_freqs):
            if np.abs(current_freq - selected_freq) < min_distance_hz:
                is_too_close = True
                # If they are too close, compare dominances
                if current_dominance > top_dominances[j]:
                    # REPLACE the existing peak with the new, stronger one
                    top_freqs[j] = current_freq
                    top_dominances[j] = current_dominance
                    top_I[j] = current_index # Update the index as well
                break # Stop checking this candidate if it conflicts
        # If it wasn't too close to ANY existing peak, add it to the lists
        if not is_too_close:
            top_freqs.append(current_freq)
            top_dominances.append(current_dominance)
            top_I.append(current_index)
        # Stop once we have gathered N items in our lists
        if len(top_freqs) >= n:
            # Trim lists to exactly N if we overshot
            top_freqs = top_freqs[:n]
            top_dominances = top_dominances[:n]
            top_I = top_I[:n]
            break

    top_freqs = np.array(top_freqs)
    top_dominances = np.array(top_dominances)

    # ---  Select last based on threshold
    b = top_dominances>dom_thres
    f0s  = top_freqs[b]
    doms = top_dominances[b]

    if plot:
        fig,axes = plt.subplots(2, 1, sharey=False, figsize=(6.4,4.8))
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20)
        #axes[0].plot(t, x, c=(0.5,0.5,0.5))
        axes[0].plot(t[n_head:], xf, c='k')
        axes[1].plot(freqs   ,S_rel)
        axes[1].plot(freqs[top_I],S_rel[top_I],'+' )
        axes[1].plot(f0s, doms,  'ko' )
        axes[1].axhline(10, c='k', ls='--')
        axes[1].set_xlabel('Frequencies ')
        axes[1].set_ylabel('Normalize spectrum')
#         ax.legend()
    return f0s, doms

def detect_periodic_by_correlation(t, x, period, rtol_corr=0.99, min_periods=3, plot=False):
    dt = t[1] - t[0]
    N = len(x)
    n_per = int(round(period / dt))
    if n_per < 4 or n_per >= N:
        return False, None, 0

    # reference = last period
    w_ref = x[-n_per:]
    w_ref = w_ref - np.mean(w_ref)
    ref_norm = np.linalg.norm(w_ref) + 1e-12

    # normalized cross-correlation
    corr = correlate(x - np.mean(x), w_ref, mode="valid")
    corr /= (ref_norm * np.sqrt(np.convolve((x - np.mean(x))**2, np.ones(n_per), mode="valid")) + 1e-12)

    # find correlation peaks
    peaks, _ = find_peaks(corr, height=rtol_corr, distance=int(0.8 * n_per))

    peaks = np.sort(peaks)
    peaks = peaks[peaks <= (len(corr) - 1)]

    good=None
    if len(peaks) >= min_periods:
        # enforce successive spacing ~ period
        good = [peaks[-1]]
        for p in reversed(peaks[:-1]):
            if abs((good[-1] - p) - n_per) <= 2:
                good.append(p)
            else:
                break

    if plot:
        print('>>> Plotting')
        plt.figure(figsize=(10, 4))
        plt.subplot(2,1,1)
        plt.plot(t, x, color='grey', label='Full signal')
        plt.plot(t[-n_per:], x[-n_per:], color='black', label='Last period')
        if good is not None:
            for g in good[:-1]:
                plt.plot(t[g:g+n_per], x[g:g+n_per], 'b--', label='Matched period' if g == good[0] else "")
        plt.legend()
        plt.title("Signal and matched periods")

        plt.subplot(2,1,2)
        plt.plot(t[:len(corr)], corr, 'r', label='Correlation')
        plt.axhline(rtol_corr, color='k', linestyle='--', label='Threshold')
        plt.title("Correlation coefficient")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()


    if len(peaks) <= min_periods:
        return False, None, 0

    if len(good) < min_periods:
        return False, None, 0

    onset_idx = good[-1]
    t_onset = t[onset_idx]
    return True, t_onset, len(good)


def detect_periodic_by_windows_overlap(t, x, period, rtol=1e-3, min_periods=3, plot=False, head_frac=0.2):
    t = np.asarray(t).copy()
    x = np.asarray(x).copy()
    N = len(x)
#     n_head = int(head_frac * N)
#     x = x[n_head:]
#     t = t[n_head:]


    dt = t[1] - t[0]
    N = len(x)
    n_per = int(round(period / dt))
    if n_per < 4 or n_per >= N:
        return False, None, 0

    n_full = N // n_per
    if n_full < min_periods:
        return False, None, 0

    # --- Reference window (over the last period)
    i_ref = np.arange(N-n_per,N)
    t_ref = t[N-n_per:N]
    x_ref = x[N-n_per:N]
    w_ref = x_ref - np.mean(x_ref)
    ref_norm = np.linalg.norm(w_ref) + 1e-12


    n_good = 1
    onset_idx = N - n_per
    good_indices = [onset_idx]
    offset = 0  # current slippage

    # Step backward over periods 
    # allow for small slippage if period is not dividable by dt
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8))
        axes[0].plot(t, x, c=(0.5,0.5,0.5), label='Signal')
        axes[0].plot(t_ref, x_ref, 'k-', label='Reference')
        axes[1].plot(t_ref, x_ref, 'k-', label='Reference')

    for k in range(1, n_full):
        best_err = np.inf
        best_i1 = None
        best_offset = None
        # Try small shifts [-1,0,1] and propagate previous offset
        for slip in [-2, -1, 0, 1, 2]:
            i1 = N - (k + 1) * n_per + offset + slip
            i2 = i1 + n_per
            if i1 < 0:
                continue
            w_curr = x[i1:i2] - np.mean(x[i1:i2])
            err = np.linalg.norm(w_curr - w_ref) / ref_norm
            if err < best_err:
                best_err = err
                best_i1 = i1
                best_offset = offset + slip

        if best_err < rtol:
            n_good += 1
            onset_idx = best_i1
            good_indices.append(best_i1)
            offset = best_offset
            if plot:
                t_curr = t[best_i1:best_i1 + n_per]
                x_curr = x[best_i1:best_i1 + n_per]
                axes[0].plot(t_curr, x_curr, '--')
                axes[1].plot(t_ref , x_curr, '--')
        else:
            break

    if n_good < min_periods:
        return False, None, 0

    if plot:
        axes[0].set_title("Windows comparison with slippage")

    return True, t[onset_idx], n_good



def detrend_lowpass(t, x, trend_wavelength_fraction=0.5, filter_order=5, plot=False):
    """
    Automatically removes slow trends from a time series (t, x) using a
    zero-phase high-pass Butterworth filter.

    Parameters:
    t (array_like): Time array (assumed to be uniformly sampled).
    x (array_like): Signal amplitude array.
    trend_wavelength_fraction (float): The desired filter cutoff is set such
        that components with wavelengths longer than this fraction of the
        total signal duration are removed.
    filter_order (int): The order of the Butterworth filter.

    Returns:
    array_like: The detrended signal.
    """
    last = x[-1]
    dt = np.mean(np.diff(t))
    sample_rate = 1.0 / dt 

    total_duration = t[-1] - t[0]
    cutoff_period = total_duration * trend_wavelength_fraction
    cutoff_frequency_hz = 1.0 / cutoff_period

    nyquist_freq = 0.5 * sample_rate
    if cutoff_frequency_hz >= nyquist_freq:
        raise ValueError("Cutoff frequency is too high for the given sample rate. Adjust the trend_wavelength_fraction.")
    normal_cutoff = cutoff_frequency_hz / nyquist_freq
    b, a = signal.butter(filter_order, normal_cutoff, btype='high', analog=False)

    x_det = signal.filtfilt(b, a, x, method="pad", padtype='odd', padlen=150)
    x_det = x_det-x_det[-1]+last

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t, x, label='Original', alpha=0.6)
        plt.plot(t, x_det, 'k-', label='Detrended', linewidth=1.5)
        plt.title('Automated Zero-Phase Detrending')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

    return x_det


if __name__ == '__main__':
    from nalulib.weio.csv_file import CSVFile


#     # 1. Generate synthetic data
#     sample_rate_gen = 100.0  # Hz
#     duration_gen = 20.0      # seconds
#     t = np.linspace(0, duration_gen, int(sample_rate_gen * duration_gen), endpoint=False)
#     # Create the components: exponential decay + sinusoid + noise
#     exponential_trend = 5.0 * np.exp(-t / 5.0)
#     sinusoid = 1.0 * np.sin(2 * np.pi * 3.0 * t) # 3 Hz sinusoid
#     x = exponential_trend + sinusoid + np.random.normal(0, 0.1, len(t))
#     x_det = zero_phase_detrend( t, x)


#     df= CSVFile('../../../nalu-cases/_results/cases_polar/du00-w-212_re03.0M/forces_aoa09.0.csv').toDataFrame()
    df= CSVFile('../../../nalu-cases/_results/cases_polar/du00-w-212_re03.0M/forces_aoa-19.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/du00-w-212_re03.0M/forces_aoa10.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/ffa-w3-211_re10.0M/forces_aoa-17.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/ffa-w3-211_re10.0M/forces_aoa19.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/ffa-w3-211_re10.0M/forces_aoa09.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/nlf1-0416_re04.0M/forces_aoa09.0.csv').toDataFrame()
#     df= CSVFile('../../../nalu-cases/_results/cases_polar/nlf1-0416_re04.0M/forces_aoa-9.0.csv').toDataFrame()
    df =df[df['Time']>0]
    t = df['Time'].values
    x = df['Fpy'].values
# 
#     f0s, doms = detect_period_by_fft_dom(t,x, plot=True, head_frac=0.2)
#     print(f0s, doms)
    c = analyze_steady_state(t, x, plot=True)
    print(c)
    plt.show()
