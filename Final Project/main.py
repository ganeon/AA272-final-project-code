import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

elevation_mask = 5

# NORTH -X; UP Z; EAST Y
def get_az_and_el(x,y,z):
    el = np.arctan2(z,np.sqrt(x**2+y**2))
    
    # East, North, Up convention
    az = np.arctan2(y, -x)
    
    # Ensure azimuth is in [0, 2π]
    if az < 0:
        az += 2 * np.pi
    
    return az, el

def create_az_arr_and_el_arr(x_arr, y_arr,z_arr):
    N = len(x_arr)
    az_arr = np.zeros(N)
    el_arr = np.zeros(N)

    for i in range(N):
        az_arr[i], el_arr[i] = get_az_and_el(x_arr[i], y_arr[i], z_arr[i])
    
    return az_arr, el_arr

def sat_in_view(x,y,z):
    if z>0:
        return True
    else:
        return False
    
def sat_in_view_el(el, elevation_mask = elevation_mask):
    if el> 2*np.pi*elevation_mask/360:
        return True
    else:
        return False

def get_unit_vector(x, y, z):
    v = np.array([x,y,z])
    return v * 1/np.linalg.norm(v)

def compute_gdop_matrix(positions):
    G = []
    for pos in positions:
        i_hat = get_unit_vector(pos[0], pos[1], pos[2])
        G_row = np.hstack((-i_hat, 1))
        G.append(G_row)
    G = np.array(G)
    H = np.linalg.inv(G.T @ G)
    return H

def compute_hdop_matrix(positions):
    G = []
    for pos in positions:
        mag = np.linalg.norm(pos)
        G_row = np.hstack((-pos[0]/mag, -pos[1]/mag, 1))
        G.append(G_row)
    G = np.array(G)
    H = np.linalg.inv(G.T @ G)
    return H

def compute_hdop_matrix_(positions):
    G = []
    for pos in positions:
        mag = np.linalg.norm(pos)
        G_row = np.hstack((-pos[0]/mag, -pos[1]/mag))
        G.append(G_row)
    G = np.array(G)
    H = np.linalg.inv(G.T @ G)
    return H

def compute_pdop(H):
    """Return PDOP = sqrt(trace(position covariance)) given H = (G^T G)^{-1}."""
    pos_cov = H[:3, :3]
    return np.sqrt(np.trace(pos_cov))

def compute_hdop(H):
    """Return HDOP = sqrt(trace(horizontal position covariance)) given H = (G^T G)^{-1}."""
    # HDOP uses only the horizontal components (X and Y, indices 0 and 1)
    horiz_cov = H[:2, :2]
    return np.sqrt(np.trace(horiz_cov))

def plot_sky_trajectories(fig_title, positions, pdop_dict, visibility_flags, sat_names, time_window=60):
    """
    Generic function to plot sky trajectories for any constellation
    
    Parameters:
    - fig_title: Title for the figure
    - positions: List of position names
    - pdop_dict: Dictionary of PDOP/HDOP values
    - visibility_flags: Dictionary of visibility flags and az/el data
    - sat_names: List of satellite names (e.g., ['D1', 'D2', 'D3', 'D4', 'MSC'])
    - time_window: Number of time steps before/after peak to plot
    """
    fig = plt.figure(figsize=(6 * len(positions), 8))
    gs = gridspec.GridSpec(1, len(positions), figure=fig, wspace=0.40, bottom=0.15)
    fig.suptitle(fig_title, fontsize=22, y=0.96)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    # Extend colors if needed for Kelly constellation (15 satellites)
    if len(sat_names) > 10:
        colors = colors + ['C0', 'C1', 'C2', 'C3', 'C4']

    # Create legend handles for ALL satellites (regardless of visibility)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=colors[i], linewidth=2.5, alpha=0.7) 
                     for i in range(len(sat_names))]
    legend_labels = sat_names

    for col, pos in enumerate(positions):
        pdop = pdop_dict[pos]
        
        if np.all(np.isnan(pdop)):
            peak_idx = None
        else:
            peak_idx = np.nanargmax(pdop)

        ax = fig.add_subplot(gs[0, col], projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(90, 0)
        ax.set_rticks([0, 30, 60, 90])
        ax.tick_params(labelsize=10)

        if peak_idx is None:
            ax.set_title(pos.replace("_", " ") + "\nNo Visible Sats", fontsize=16)
            continue

        # Load visibility data
        flags = visibility_flags[pos]
        flag_list = [flags[f"f{i+1}"] for i in range(len(sat_names))]
        az_list = flags["az"]
        el_list = flags["el"]

        # Define time window
        N = len(flag_list[0])
        start_idx = max(0, peak_idx - time_window)
        end_idx = min(N, peak_idx + time_window)

        # Plot trajectories - ONLY when satellite is visible
        for sat_idx, (az_arr, el_arr, flg, color, sat_name) in enumerate(
            zip(az_list, el_list, flag_list, colors, sat_names)
        ):
            # Work with indices in the time window
            visible_mask = flg[start_idx:end_idx] == 1
            
            if not np.any(visible_mask):
                continue
            
            # Get actual indices where satellite is visible
            time_indices = np.arange(start_idx, end_idx)
            visible_time_indices = time_indices[visible_mask]
            
            # Find continuous segments (satellite sets/rises create breaks)
            time_breaks = np.where(np.diff(visible_time_indices) > 1)[0] + 1
            continuous_segments = np.split(visible_time_indices, time_breaks)
            
            # Plot each continuous segment
            for segment_idx in continuous_segments:
                if len(segment_idx) < 2:
                    continue
                
                seg_az = az_arr[segment_idx]
                seg_el = el_arr[segment_idx]
                
                
                # No large jumps, plot entire segment
                r = np.degrees(seg_el)
                ax.plot(seg_az, r, linewidth=2.5, alpha=0.7, color=color)

        # Plot satellite positions at peak PDOP (only if visible)
        sats_az = []
        sats_el = []
        sats_colors = []
        
        for az_arr, el_arr, flg, color, sat_name in zip(az_list, el_list, flag_list, colors, sat_names):
            if flg[peak_idx] == 1:  # Only if satellite is visible at peak
                sats_az.append(az_arr[peak_idx])
                sats_el.append(el_arr[peak_idx])
                sats_colors.append(color)

        if len(sats_az) > 0:
            ax.scatter(sats_az, np.degrees(np.array(sats_el)), 
                      s=200, c=sats_colors, edgecolors="black", linewidth=2.5, zorder=5, marker='o')

        ax.set_title(f"{pos.replace('_',' ')}\nPeak HDOP = {pdop[peak_idx]:.2f}",
                    fontsize=16, pad=14)

    # Add legend at the bottom of the figure with ALL satellites
    fig.legend(legend_handles, legend_labels, loc='lower center', 
              ncol=min(len(legend_labels), 8), fontsize=11, 
              frameon=True, fancybox=True, shadow=True)

    plt.show()


positions = ["Hellas_Planitia", "Valles_Marineris", "Schiaparelli_Crater", "Mawrth_Vallis", "Arcadia_Planitia"]

##################################################################################################################
# CONSTELLATION A
##################################################################################################################

pdop_dict = {}
hdop_dict = {}  # Added HDOP dictionary
sat_count_dict = {}
visibility_flags = {}

for pos in positions:
    csv1 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationA/D1_A_Coords.csv")
    x1 = csv1[f"D1.{pos}_GS.X"].values
    y1 = csv1[f"D1.{pos}_GS.Y"].values
    z1 = csv1[f"D1.{pos}_GS.Z"].values

    csv2 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationA/D2_A_Coords.csv")
    x2 = csv2[f"D2.{pos}_GS.X"].values
    y2 = csv2[f"D2.{pos}_GS.Y"].values
    z2 = csv2[f"D2.{pos}_GS.Z"].values

    csv3 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationA/D3_A_Coords.csv")
    x3 = csv3[f"D3.{pos}_GS.X"].values
    y3 = csv3[f"D3.{pos}_GS.Y"].values
    z3 = csv3[f"D3.{pos}_GS.Z"].values

    csv4 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationA/D4_A_Coords.csv")
    x4 = csv4[f"D4.{pos}_GS.X"].values
    y4 = csv4[f"D4.{pos}_GS.Y"].values
    z4 = csv4[f"D4.{pos}_GS.Z"].values

    csv5 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationA/MSC_A_Coords.csv")
    x5 = csv5[f"MSC.{pos}_GS.X"].values
    y5 = csv5[f"MSC.{pos}_GS.Y"].values
    z5 = csv5[f"MSC.{pos}_GS.Z"].values

    N = len(x1)
    f1 = np.zeros(N)
    f2 = np.zeros(N)
    f3 = np.zeros(N)
    f4 = np.zeros(N)
    f5 = np.zeros(N)

    az_arr1, el_arr1 = create_az_arr_and_el_arr(x1,y1,z1)
    az_arr2, el_arr2 = create_az_arr_and_el_arr(x2,y2,z2)
    az_arr3, el_arr3 = create_az_arr_and_el_arr(x3,y3,z3)
    az_arr4, el_arr4 = create_az_arr_and_el_arr(x4,y4,z4)
    az_arr5, el_arr5 = create_az_arr_and_el_arr(x5,y5,z5)

    for i in range(N):
        if sat_in_view_el(el_arr1[i]): f1[i] = 1
        if sat_in_view_el(el_arr2[i]): f2[i] = 1
        if sat_in_view_el(el_arr3[i]): f3[i] = 1
        if sat_in_view_el(el_arr4[i]): f4[i] = 1
        if sat_in_view_el(el_arr5[i]): f5[i] = 1

    sat_count = f1 + f2 + f3 + f4 + f5
    sat_count_dict[pos] = sat_count

    visibility_flags[pos] = {
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
        "az": (az_arr1, az_arr2, az_arr3, az_arr4, az_arr5),
        "el": (el_arr1, el_arr2, el_arr3, el_arr4, el_arr5)
    }

    PDOP_arr = np.zeros(N)
    HDOP_arr = np.zeros(N)  # Added HDOP array
    for i in range(N):
        visible_count = f1[i] + f2[i] + f3[i] + f4[i] + f5[i]
        
        # HDOP requires 3+ satellites, PDOP requires 4+
        if visible_count >= 3:
            positions_ = []
            if f1[i]==1: positions_.append((x1[i], y1[i], z1[i]))
            if f2[i]==1: positions_.append((x2[i], y2[i], z2[i]))
            if f3[i]==1: positions_.append((x3[i], y3[i], z3[i]))
            if f4[i]==1: positions_.append((x4[i], y4[i], z4[i]))
            if f5[i]==1: positions_.append((x5[i], y5[i], z5[i]))
            positions_ = np.array(positions_)
            H = compute_hdop_matrix(positions=positions_)
            HDOP_arr[i] = compute_hdop(H)  # Compute HDOP with 3+ sats
            
            if visible_count >= 4:
                H = compute_gdop_matrix(positions=positions_)
                PDOP_arr[i] = compute_pdop(H)  # Compute PDOP only with 4+ sats
            else:
                PDOP_arr[i] = np.nan
        else:
            PDOP_arr[i] = np.nan
            HDOP_arr[i] = np.nan

    pdop_dict[pos] = PDOP_arr
    hdop_dict[pos] = HDOP_arr  # Store HDOP

    print(f"Constellation A average PDOP for {pos}:{np.nanmean(PDOP_arr)}")
    print(f"Constellation A percetage PDOP uptime {pos}:{np.sum(~np.isnan(PDOP_arr)) / PDOP_arr.size}")
    print(f"Constellation A average HDOP for {pos}:{np.nanmean(HDOP_arr)}")
    print(f"Constellation A percetage HDOP uptime {pos}:{np.sum(~np.isnan(HDOP_arr)) / HDOP_arr.size}")

# ----- PLOTTING PDOP, HDOP AND SATELLITE COUNT SIDE BY SIDE -----
fig, axes = plt.subplots(
    2,
    len(positions),
    figsize=(5 * len(positions), 8),
    constrained_layout=True
)

fig.suptitle("PDOP/HDOP and Satellite Visibility — Constellation A", fontsize=16)

for col, pos in enumerate(positions):
    # Top row: PDOP and HDOP
    ax_pdop = axes[0, col]
    pdop = pdop_dict[pos]
    hdop = hdop_dict[pos]

    ax_pdop.plot(pdop, label='PDOP', color='blue', linewidth=1.5)
    ax_pdop.plot(hdop, label='HDOP', color='green', linewidth=1.5, linestyle='--')

    # Red crosses only when BOTH PDOP and HDOP are NaN (fewer than 3 satellites)
    both_nan_idx = np.where(np.isnan(pdop) & np.isnan(hdop))[0]
    if len(both_nan_idx) > 0:
        ax_pdop.scatter(both_nan_idx, np.zeros_like(both_nan_idx), marker='x', color='red', s=40)

    if np.any(~np.isnan(pdop)):
        # Peak PDOP
        peak_idx_pdop = np.nanargmax(pdop)
        peak_val_pdop = pdop[peak_idx_pdop]
        ax_pdop.scatter(peak_idx_pdop, peak_val_pdop, marker='o', color='blue', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_pdop, peak_val_pdop, f"P:{peak_val_pdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='blue')
        
        # Peak HDOP
        peak_idx_hdop = np.nanargmax(hdop)
        peak_val_hdop = hdop[peak_idx_hdop]
        ax_pdop.scatter(peak_idx_hdop, peak_val_hdop, marker='s', color='green', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_hdop, peak_val_hdop, f"H:{peak_val_hdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='green')

    ax_pdop.set_title(f"{pos}")
    ax_pdop.grid(True)
    ax_pdop.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_pdop.set_ylabel("DOP")

    # Bottom row: Satellite count
    ax_sat = axes[1, col]
    sat_count = sat_count_dict[pos]

    ax_sat.plot(sat_count, linewidth=2, color='C0')
    ax_sat.fill_between(range(len(sat_count)), sat_count, alpha=0.3)
    
    ax_sat.axhline(y=4, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Min for PDOP (4)')
    ax_sat.axhline(y=3, color='green', linestyle=':', linewidth=1, alpha=0.7, label='Min for HDOP (3)')

    ax_sat.set_xlabel("Time Index")
    ax_sat.set_ylim([0, 5.5])
    ax_sat.set_yticks(range(6))
    ax_sat.grid(True, alpha=0.3)
    ax_sat.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_sat.set_ylabel("Number of Satellites")

plt.show()



# Plot Constellation A
plot_sky_trajectories(
    "Constellation A — Sky Plots at Peak PDOP (with Satellite Paths)",
    positions,
    pdop_dict,
    visibility_flags,
    sat_names=['D1', 'D2', 'D3', 'D4', 'MSC'],
    time_window=20  # Reduced for cleaner trajectories
)

##################################################################################################################
# CONSTELLATION B
##################################################################################################################

pdop_dict_B = {}
hdop_dict_B = {}  # Added HDOP dictionary
sat_count_dict_B = {}
visibility_flags_B = {}

for pos in positions:
    csv1 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationB/D1_B_Coords.csv")
    x1 = csv1[f"D1.{pos}_GS.X"].values
    y1 = csv1[f"D1.{pos}_GS.Y"].values
    z1 = csv1[f"D1.{pos}_GS.Z"].values

    csv2 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationB/D2_B_Coords.csv")
    x2 = csv2[f"D2.{pos}_GS.X"].values
    y2 = csv2[f"D2.{pos}_GS.Y"].values
    z2 = csv2[f"D2.{pos}_GS.Z"].values

    csv3 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationB/D3_B_Coords.csv")
    x3 = csv3[f"D3.{pos}_GS.X"].values
    y3 = csv3[f"D3.{pos}_GS.Y"].values
    z3 = csv3[f"D3.{pos}_GS.Z"].values

    csv4 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationB/D4_B_Coords.csv")
    x4 = csv4[f"D4.{pos}_GS.X"].values
    y4 = csv4[f"D4.{pos}_GS.Y"].values
    z4 = csv4[f"D4.{pos}_GS.Z"].values

    csv5 = pd.read_csv("SatelliteCoordinatesData/Molli/ConstellationB/MSC_B_Coords.csv")
    x5 = csv5[f"MSC.{pos}_GS.X"].values
    y5 = csv5[f"MSC.{pos}_GS.Y"].values
    z5 = csv5[f"MSC.{pos}_GS.Z"].values

    N = len(x1)
    f1 = np.zeros(N)
    f2 = np.zeros(N)
    f3 = np.zeros(N)
    f4 = np.zeros(N)
    f5 = np.zeros(N)

    az_arr1, el_arr1 = create_az_arr_and_el_arr(x1,y1,z1)
    az_arr2, el_arr2 = create_az_arr_and_el_arr(x2,y2,z2)
    az_arr3, el_arr3 = create_az_arr_and_el_arr(x3,y3,z3)
    az_arr4, el_arr4 = create_az_arr_and_el_arr(x4,y4,z4)
    az_arr5, el_arr5 = create_az_arr_and_el_arr(x5,y5,z5)

    for i in range(N):
        if sat_in_view_el(el_arr1[i]): f1[i] = 1
        if sat_in_view_el(el_arr2[i]): f2[i] = 1
        if sat_in_view_el(el_arr3[i]): f3[i] = 1
        if sat_in_view_el(el_arr4[i]): f4[i] = 1
        if sat_in_view_el(el_arr5[i]): f5[i] = 1

    sat_count = f1 + f2 + f3 + f4 + f5
    sat_count_dict_B[pos] = sat_count

    visibility_flags_B[pos] = {
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
        "az": (az_arr1, az_arr2, az_arr3, az_arr4, az_arr5),
        "el": (el_arr1, el_arr2, el_arr3, el_arr4, el_arr5)
    }

    PDOP_arr = np.zeros(N)
    HDOP_arr = np.zeros(N)  # Added HDOP array
    for i in range(N):
        visible_count = f1[i] + f2[i] + f3[i] + f4[i] + f5[i]
        
        # HDOP requires 3+ satellites, PDOP requires 4+
        if visible_count >= 3:
            positions_ = []
            if f1[i]==1: positions_.append((x1[i], y1[i], z1[i]))
            if f2[i]==1: positions_.append((x2[i], y2[i], z2[i]))
            if f3[i]==1: positions_.append((x3[i], y3[i], z3[i]))
            if f4[i]==1: positions_.append((x4[i], y4[i], z4[i]))
            if f5[i]==1: positions_.append((x5[i], y5[i], z5[i]))
            positions_ = np.array(positions_)
            H = compute_hdop_matrix(positions=positions_)
            HDOP_arr[i] = compute_hdop(H)  # Compute HDOP with 3+ sats
            
            if visible_count >= 4:
                H = compute_gdop_matrix(positions=positions_)
                PDOP_arr[i] = compute_pdop(H)  # Compute PDOP only with 4+ sats
            else:
                PDOP_arr[i] = np.nan
        else:
            PDOP_arr[i] = np.nan
            HDOP_arr[i] = np.nan

    pdop_dict_B[pos] = PDOP_arr
    hdop_dict_B[pos] = HDOP_arr  # Store HDOP

    print(f"Constellation B average PDOP for {pos}:{np.nanmean(PDOP_arr)}")
    print(f"Constellation B percetage PDOP uptime {pos}:{np.sum(~np.isnan(PDOP_arr)) / PDOP_arr.size}")
    print(f"Constellation B average HDOP for {pos}:{np.nanmean(HDOP_arr)}")
    print(f"Constellation B percetage PDOP uptime {pos}:{np.sum(~np.isnan(HDOP_arr)) / HDOP_arr.size}")


# ----- PLOTTING PDOP, HDOP AND SATELLITE COUNT SIDE BY SIDE -----
fig, axes = plt.subplots(
    2,
    len(positions),
    figsize=(5 * len(positions), 8),
    constrained_layout=True
)

fig.suptitle("PDOP/HDOP and Satellite Visibility — Constellation B", fontsize=16)

for col, pos in enumerate(positions):
    # Top row: PDOP and HDOP
    ax_pdop = axes[0, col]
    pdop = pdop_dict_B[pos]
    hdop = hdop_dict_B[pos]

    ax_pdop.plot(pdop, label='PDOP', color='blue', linewidth=1.5)
    ax_pdop.plot(hdop, label='HDOP', color='green', linewidth=1.5, linestyle='--')

    # Red crosses only when BOTH PDOP and HDOP are NaN (fewer than 3 satellites)
    both_nan_idx = np.where(np.isnan(pdop) & np.isnan(hdop))[0]
    if len(both_nan_idx) > 0:
        ax_pdop.scatter(both_nan_idx, np.zeros_like(both_nan_idx), marker='x', color='red', s=40, label='No DOP')

    if np.any(~np.isnan(pdop)):
        # Peak PDOP
        peak_idx_pdop = np.nanargmax(pdop)
        peak_val_pdop = pdop[peak_idx_pdop]
        ax_pdop.scatter(peak_idx_pdop, peak_val_pdop, marker='o', color='blue', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_pdop, peak_val_pdop, f"P:{peak_val_pdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='blue')
        
        # Peak HDOP
        peak_idx_hdop = np.nanargmax(hdop)
        peak_val_hdop = hdop[peak_idx_hdop]
        ax_pdop.scatter(peak_idx_hdop, peak_val_hdop, marker='s', color='green', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_hdop, peak_val_hdop, f"H:{peak_val_hdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='green')

    ax_pdop.set_title(f"{pos}")
    ax_pdop.grid(True)
    ax_pdop.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_pdop.set_ylabel("DOP")

    # Bottom row: Satellite count
    ax_sat = axes[1, col]
    sat_count = sat_count_dict_B[pos]

    ax_sat.plot(sat_count, linewidth=2, color='C0')
    ax_sat.fill_between(range(len(sat_count)), sat_count, alpha=0.3)
    
    ax_sat.axhline(y=4, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Min for PDOP (4)')
    ax_sat.axhline(y=3, color='green', linestyle=':', linewidth=1, alpha=0.7, label='Min for HDOP (3)')

    ax_sat.set_xlabel("Time Index")
    ax_sat.set_ylim([0, 5.5])
    ax_sat.set_yticks(range(6))
    ax_sat.grid(True, alpha=0.3)
    ax_sat.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_sat.set_ylabel("Number of Satellites")

plt.show()

# Plot Constellation B
plot_sky_trajectories(
    "Constellation B — Sky Plots at Peak PDOP (with Satellite Paths)",
    positions,
    pdop_dict_B,
    visibility_flags_B,
    sat_names=['D1', 'D2', 'D3', 'D4', 'MSC'],
    time_window=20  # Reduced for cleaner trajectories
)

##################################################################################################################
# KELLY CONSTELLATION
##################################################################################################################

satellites = [f"{i}_{j}" for i in range(1, 6) for j in range(1, 4)]
pdop_dict_kelly = {}
hdop_dict_kelly = {}  # Added HDOP dictionary
sat_count_dict_kelly = {}
visibility_flags_kelly = {}

for pos in positions:
    coords = {}
    for sat in satellites:
        df = pd.read_csv(f"SatelliteCoordinatesData/Kelly/COMPASS_{sat}_Coords.csv")
        coords[sat] = {
            "x": df[f"COMPASS_{sat}.{pos}_GS.X"].values,
            "y": df[f"COMPASS_{sat}.{pos}_GS.Y"].values,
            "z": df[f"COMPASS_{sat}.{pos}_GS.Z"].values,
        }

    N = len(next(iter(coords.values()))["x"])
    flags = {sat: np.zeros(N) for sat in satellites}
    azel = {}

    for sat in satellites:
        x = coords[sat]["x"]
        y = coords[sat]["y"]
        z = coords[sat]["z"]
        az_arr, el_arr = create_az_arr_and_el_arr(x, y, z)
        azel[sat] = (az_arr, el_arr)
        
        for i in range(N):
            if sat_in_view_el(el=el_arr[i]):
                flags[sat][i] = 1

    sat_count = np.sum([flags[sat] for sat in satellites], axis=0)
    sat_count_dict_kelly[pos] = sat_count

    # Store visibility flags in the format expected by plot function
    visibility_flags_kelly[pos] = {
        f"f{i+1}": flags[satellites[i]] for i in range(15)
    }
    visibility_flags_kelly[pos]["az"] = tuple(azel[sat][0] for sat in satellites)
    visibility_flags_kelly[pos]["el"] = tuple(azel[sat][1] for sat in satellites)

    PDOP_arr = np.full(N, np.nan)
    HDOP_arr = np.full(N, np.nan)  # Added HDOP array
    for i in range(N):
        visible_sats = [sat for sat in satellites if flags[sat][i] == 1]
        
        # HDOP requires 3+ satellites, PDOP requires 4+
        if len(visible_sats) >= 3:
            positions_ = [(coords[sat]["x"][i], coords[sat]["y"][i], coords[sat]["z"][i]) 
                         for sat in visible_sats]
            H = compute_hdop_matrix(positions=positions_)
            HDOP_arr[i] = compute_hdop(H)  # Compute HDOP with 3+ sats
            
            if len(visible_sats) >= 4:
                H = compute_gdop_matrix(positions=positions_)
                PDOP_arr[i] = compute_pdop(H)  # Compute PDOP only with 4+ sats

    pdop_dict_kelly[pos] = PDOP_arr
    hdop_dict_kelly[pos] = HDOP_arr  # Store HDOP

    print(f"Kelly average PDOP for {pos}:{np.nanmean(PDOP_arr)}")
    print(f"Kelly percetage PDOP uptime {pos}:{np.sum(~np.isnan(PDOP_arr)) / PDOP_arr.size}")
    print(f"Kelly average HDOP for {pos}:{np.nanmean(HDOP_arr)}")
    print(f"Kelly percetage HDOP uptime {pos}:{np.sum(~np.isnan(HDOP_arr)) / HDOP_arr.size}")

# ----- PLOTTING PDOP, HDOP AND SATELLITE COUNT SIDE BY SIDE -----
fig, axes = plt.subplots(
    2,
    len(positions),
    figsize=(5 * len(positions), 8),
    constrained_layout=True
)

fig.suptitle("PDOP/HDOP and Satellite Visibility — Kelly Constellation", fontsize=16)

for col, pos in enumerate(positions):
    # Top row: PDOP and HDOP
    ax_pdop = axes[0, col]
    pdop = pdop_dict_kelly[pos]
    hdop = hdop_dict_kelly[pos]

    ax_pdop.plot(pdop, label='PDOP', color="blue", linewidth=1.5)
    ax_pdop.plot(hdop, label='HDOP', color="green", linewidth=1.5, linestyle='--')

    # Red crosses only when BOTH PDOP and HDOP are NaN (fewer than 3 satellites)
    both_nan_idx = np.where(np.isnan(pdop) & np.isnan(hdop))[0]
    if len(both_nan_idx) > 0:
        finite_vals = hdop[~np.isnan(hdop)]
        baseline = np.min(finite_vals) - 0.2 if len(finite_vals) > 0 else 0
        ax_pdop.scatter(both_nan_idx, np.full_like(both_nan_idx, baseline, dtype=float), marker='x', color='red', s=40, label='No DOP')

    if np.any(~np.isnan(pdop)):
        # Peak PDOP
        peak_idx_pdop = np.nanargmax(pdop)
        peak_val_pdop = pdop[peak_idx_pdop]
        ax_pdop.scatter(peak_idx_pdop, peak_val_pdop, marker='o', color='blue', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_pdop, peak_val_pdop, f"P:{peak_val_pdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='blue')
        
        # Peak HDOP
        peak_idx_hdop = np.nanargmax(hdop)
        peak_val_hdop = hdop[peak_idx_hdop]
        ax_pdop.scatter(peak_idx_hdop, peak_val_hdop, marker='s', color='green', s=60, edgecolors='black', zorder=5)
        ax_pdop.text(peak_idx_hdop, peak_val_hdop, f"H:{peak_val_hdop:.2f}",
                fontsize=9, ha='center', va='bottom', color='green')

    ax_pdop.set_title(f"{pos}")
    ax_pdop.grid(True)
    ax_pdop.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_pdop.set_ylabel("DOP")

    # Bottom row: Satellite count
    ax_sat = axes[1, col]
    sat_count = sat_count_dict_kelly[pos]

    ax_sat.plot(sat_count, linewidth=2, color='C0')
    ax_sat.fill_between(range(len(sat_count)), sat_count, alpha=0.3)
    
    ax_sat.axhline(y=4, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Min for PDOP (4)')
    ax_sat.axhline(y=3, color='green', linestyle=':', linewidth=1, alpha=0.7, label='Min for HDOP (3)')

    ax_sat.set_xlabel("Time Index")
    ax_sat.set_ylim([0, max(16, np.max(sat_count) + 1)])
    ax_sat.grid(True, alpha=0.3)
    ax_sat.legend(loc='upper right', fontsize=8)
    if col == 0:
        ax_sat.set_ylabel("Number of Satellites")

plt.show()

# Plot Kelly
plot_sky_trajectories(
    "Kelly Constellation — Sky Plots at Peak PDOP (with Satellite Paths)",
    positions,
    pdop_dict_kelly,
    visibility_flags_kelly,
    sat_names=[f"{i}_{j}" for i in range(1, 6) for j in range(1, 4)],
    time_window=20  # Reduced for cleaner trajectories
)

##################################################################################################################
# IIYAMA CONSTELLATION (formerly NAV LAB)
##################################################################################################################

pdop_dict_iiyama = {}
sat_count_dict_iiyama = {}
visibility_flags_iiyama = {}

for pos in positions:
    csv1 = pd.read_csv("SatelliteCoordinatesData/NAV LAB/AS1_Coords.csv")
    x1 = csv1[f"AS1.{pos}_GS.X"].values
    y1 = csv1[f"AS1.{pos}_GS.Y"].values
    z1 = csv1[f"AS1.{pos}_GS.Z"].values

    csv2 = pd.read_csv("SatelliteCoordinatesData/NAV LAB/AS2_Coords.csv")
    x2 = csv2[f"AS2.{pos}_GS.X"].values
    y2 = csv2[f"AS2.{pos}_GS.Y"].values
    z2 = csv2[f"AS2.{pos}_GS.Z"].values

    csv3 = pd.read_csv("SatelliteCoordinatesData/NAV LAB/RGT_1_Coords.csv")
    x3 = csv3[f"RGT_1.{pos}_GS.X"].values
    y3 = csv3[f"RGT_1.{pos}_GS.Y"].values
    z3 = csv3[f"RGT_1.{pos}_GS.Z"].values

    csv4 = pd.read_csv("SatelliteCoordinatesData/NAV LAB/RGT_2_Coords.csv")
    x4 = csv4[f"RGT_2.{pos}_GS.X"].values
    y4 = csv4[f"RGT_2.{pos}_GS.Y"].values
    z4 = csv4[f"RGT_2.{pos}_GS.Z"].values

    csv5 = pd.read_csv("SatelliteCoordinatesData/NAV LAB/RGT_3_Coords.csv")
    x5 = csv5[f"RGT_3.{pos}_GS.X"].values
    y5 = csv5[f"RGT_3.{pos}_GS.Y"].values
    z5 = csv5[f"RGT_3.{pos}_GS.Z"].values

    N = len(x1)
    f = np.zeros((5, N))

    az_arr1, el_arr1 = create_az_arr_and_el_arr(x1, y1, z1)
    az_arr2, el_arr2 = create_az_arr_and_el_arr(x2, y2, z2)
    az_arr3, el_arr3 = create_az_arr_and_el_arr(x3, y3, z3)
    az_arr4, el_arr4 = create_az_arr_and_el_arr(x4, y4, z4)
    az_arr5, el_arr5 = create_az_arr_and_el_arr(x5, y5, z5)

    az_arrays = [az_arr1, az_arr2, az_arr3, az_arr4, az_arr5]
    el_arrays = [el_arr1, el_arr2, el_arr3, el_arr4, el_arr5]
    xyz_arrays = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4), (x5, y5, z5)]

    for s in range(5):
        for i in range(N):
            if sat_in_view_el(el_arrays[s][i]):
                f[s, i] = 1

    visibility_flags_iiyama[pos] = {
        "f1": f[0, :], "f2": f[1, :], "f3": f[2, :], "f4": f[3, :], "f5": f[4, :],
        "az": az_arrays,
        "el": el_arrays
    }

    sat_count = np.sum(f, axis=0)
    sat_count_dict_iiyama[pos] = sat_count

    PDOP_arr = np.zeros(N)
    for i in range(N):
        visible_sats = np.where(f[:, i] == 1)[0]
        if len(visible_sats) < 2:
            PDOP_arr[i] = np.nan
            continue
        positions_ = []
        for s in visible_sats:
            x_s, y_s, z_s = xyz_arrays[s]
            positions_.append([x_s[i], y_s[i], z_s[i]])
        positions_ = np.array(positions_)
        H = compute_hdop_matrix_(positions=positions_)
        PDOP_arr[i] = compute_hdop(H)

    pdop_dict_iiyama[pos] = PDOP_arr

    print(f"Iiyama average HDOP for {pos}:{np.nanmean(PDOP_arr)}")
    print(f"Iiyama percetage HDOP uptime {pos}:{np.sum(~np.isnan(PDOP_arr)) / PDOP_arr.size}")

# ----- PLOTTING HDOP AND SATELLITE COUNT SIDE BY SIDE -----
fig, axes = plt.subplots(
    2,
    len(positions),
    figsize=(5 * len(positions), 8),
    constrained_layout=True
)

fig.suptitle("HDOP and Satellite Visibility — Iiyama Constellation", fontsize=16)

for col, pos in enumerate(positions):
    # Top row: HDOP
    ax_pdop = axes[0, col]
    pdop = pdop_dict_iiyama[pos]

    ax_pdop.plot(pdop, color="blue", linewidth=1.5)

    nan_idx = np.where(np.isnan(pdop))[0]
    if len(nan_idx) > 0:
        finite = pdop[~np.isnan(pdop)]
        baseline = (np.min(finite) - 0.3) if len(finite) > 0 else -0.5
        ax_pdop.scatter(nan_idx, np.full_like(nan_idx, baseline), marker='x', color='red', s=40)

    if np.any(~np.isnan(pdop)):
        peak_idx = np.nanargmax(pdop)
        peak_val = pdop[peak_idx]
        ax_pdop.scatter(peak_idx, peak_val, color="black", s=70)
        ax_pdop.text(peak_idx, peak_val, f"{peak_val:.2f}", fontsize=9, ha="center", va="bottom")

    ax_pdop.set_title(pos)
    ax_pdop.grid(True)
    if col == 0:
        ax_pdop.set_ylabel("HDOP")

    # Bottom row: Satellite count
    ax_sat = axes[1, col]
    sat_count = sat_count_dict_iiyama[pos]
    
    ax_sat.plot(sat_count, linewidth=2)
    ax_sat.fill_between(range(len(sat_count)), sat_count, alpha=0.3)
    ax_sat.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax_sat.set_xlabel("Time Index")
    ax_sat.set_ylim([0, 5.5])
    ax_sat.set_yticks(range(6))
    ax_sat.grid(True, alpha=0.3)
    if col == 0:
        ax_sat.set_ylabel("Number of Satellites")

plt.show()

# Plot Iiyama
plot_sky_trajectories(
    "Iiyama Constellation — Sky Plots at Peak HDOP (with Satellite Paths)",
    positions,
    pdop_dict_iiyama,
    visibility_flags_iiyama,
    sat_names=['AS1', 'AS2', 'RGT1', 'RGT2', 'RGT3'],
    time_window=20  # Reduced for cleaner trajectories
)