"""
Sprint 1 - Generate All Figures for Paper/Report (US6)
Reads outputs/tables/results_resolution_metrics.csv and outputs 4 high-res plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. Setup Input & Output Paths
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "outputs" / "tables" / "results_resolution_metrics.csv"
    out_dir = root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True) 

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find CSV at {csv_path}. Please run US3 script first.")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Enforce decreasing pixel count order (High → Low)
    res_order = ["1920x1080", "1280x720", "854x480"]
    df['resolution'] = pd.Categorical(df['resolution'], categories=res_order, ordered=True)
    df = df.sort_values('resolution')

    # Helper function for common plot styling
    def setup_plot(title, ylabel):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Resolution (High → Low)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig, ax

    # ==========================================
    # Figure 1: RMSE (Full + ROI) vs Resolution
    # ==========================================
    fig1, ax1 = setup_plot('Figure 1: RMSE Degradation vs Resolution', 'RMSE Error ↓')
    ax1.plot(df['resolution'], df['rmse_full'], marker='o', color='#d62728', label='Full-frame RMSE', linewidth=2.5, markersize=8)
    ax1.plot(df['resolution'], df['rmse_roi'], marker='s', linestyle='--', color='#ff7f0e', label='ROI RMSE', linewidth=2.5, markersize=8)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / 'fig1_rmse_vs_resolution.png', dpi=300)
    print("✅ Generated: Figure 1 (RMSE)")

    # ==========================================
    # Figure 2: AbsRel (Full + ROI) vs Resolution
    # ==========================================
    fig2, ax2 = setup_plot('Figure 2: AbsRel Degradation vs Resolution', 'Absolute Relative Error (AbsRel) ↓')
    ax2.plot(df['resolution'], df['absrel_full'], marker='^', color='#9467bd', label='Full-frame AbsRel', linewidth=2.5, markersize=8)
    ax2.plot(df['resolution'], df['absrel_roi'], marker='D', linestyle='--', color='#8c564b', label='ROI AbsRel', linewidth=2.5, markersize=8)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / 'fig2_absrel_vs_resolution.png', dpi=300)
    print("✅ Generated: Figure 2 (AbsRel)")

    # ==========================================
    # Figure 3: FPS vs Resolution
    # ==========================================
    fig3, ax3 = setup_plot('Figure 3: Inference Speed vs Resolution', 'Frames Per Second (FPS) ↑')
    ax3.plot(df['resolution'], df['fps'], marker='X', color='#1f77b4', label='RTX 2050 FPS', linewidth=2.5, markersize=10)
    ax3.axhline(y=15, color='gray', linestyle=':', linewidth=2, label='15 FPS Threshold')
    ax3.set_ylim(0, 18)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / 'fig3_fps_vs_resolution.png', dpi=300)
    print("✅ Generated: Figure 3 (FPS)")

    # ==========================================
    # Figure 4: Consistency vs Speed (Pareto)
    # ==========================================
    fig4, ax4a = plt.subplots(figsize=(9, 6))
    color_err = '#d62728'
    color_fps = '#1f77b4'

    # Left Y-Axis: Error
    ax4a.set_xlabel('Resolution (High → Low)', fontsize=12, fontweight='bold')
    ax4a.set_ylabel('RMSE Error (ROI) ↓', color=color_err, fontsize=12, fontweight='bold')
    line1, = ax4a.plot(df['resolution'], df['rmse_roi'], marker='s', linestyle='--', color=color_err, label='ROI RMSE', linewidth=2.5, markersize=8)
    ax4a.tick_params(axis='y', labelcolor=color_err)

    # Right Y-Axis: FPS
    ax4b = ax4a.twinx()
    ax4b.set_ylabel('Inference Speed (FPS) ↑', color=color_fps, fontsize=12, fontweight='bold')
    line2, = ax4b.plot(df['resolution'], df['fps'], marker='^', color=color_fps, label='RTX 2050 FPS', linewidth=2.5, markersize=9)
    line3 = ax4b.axhline(y=15, color='gray', linestyle=':', linewidth=2, label='15 FPS Threshold')
    ax4b.tick_params(axis='y', labelcolor=color_fps)
    ax4b.set_ylim(0, 18)

    # Annotate Pareto Optimal Candidate
    ax4a.annotate('Pareto Optimal\n(Zero Error, Max Valid FPS)', 
                 xy=(0, df['rmse_roi'].iloc[0]), 
                 xytext=(0.3, df['rmse_roi'].iloc[-1]*0.5), 
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    # Merge legends from both axes
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax4a.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.title('Figure 4: Consistency vs. Speed Trade-off (Pareto)', fontsize=14, fontweight='bold')
    ax4a.grid(True, linestyle='--', alpha=0.6)
    fig4.tight_layout()
    
    fig4.savefig(out_dir / 'fig4_pareto_tradeoff.png', dpi=300)
    print("✅ Generated: Figure 4 (Pareto Trade-off)")
    print(f"\n🎉 All figures saved successfully to: {out_dir}")

if __name__ == "__main__":
    main()
