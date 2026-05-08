#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ====== 项目路径 ======
project_root = Path.cwd()
print(f"项目根路径: {project_root}")

# ====== 统一配置区（面向表格展示） ======
MODEL_SETS = {
    # 表1：消融对比（用 Proposed 代替 HFnet）
    "table1": {
        "display_names": {
            "VCTK": {
                "clean": "Clean",
                "noisy": "Noisy",
                "FasNet": "FasNet",
                "FasNet-TAC": "FasNet-TAC",
                "TFSkiMNet": "TFSkiMNet",
                "HGTCRN": "HGTCRN",
                "Proposed_light": "Proposed_light",
                "Proposed_base": "Proposed_base",
            },
        },
        "order": {
            "VCTK": ["clean", "noisy", "FasNet", "FasNet-TAC", "TFSkiMNet", "HGTCRN", "Proposed_light", "Proposed_base"],
        },
    },
    # 表2：结构消融
    "table2": {
        "display_names": {
            "ablation": {
                "a_YLMS_IVA_SN": "(a) Y_LMS + IVA-S&N_LMS",
                "b_YReIm_IVA_SN": "(b) Y_Re&Im + IVA-S&N_LMS",
                "c_YReIm": "(c) Y_Re&Im",
                "d_YLMS_IVA_S": "(d) Y_LMS + IVA-S_LMS",
                "e_YLMS": "(e) Y_LMS",
                "f_wo_CBAM": "(f) w/o CBAM",
                "g_wo_Mel": "(g) w/o Mel-domain",
                "h_wo_Asym": "(h) w/o Asymmetric",
            },
        },
        "order": {
            "ablation": ["a_YLMS_IVA_SN", "b_YReIm_IVA_SN", "c_YReIm", "d_YLMS_IVA_S", "e_YLMS", "f_wo_CBAM", "g_wo_Mel", "h_wo_Asym"],
        },
    },
}

# 数据集和信噪比配置
datasets = ["VCTK"]
snr_mapping = {
    "VCTK": ["N5", "0", "5"],
}
snr_display = {"N5": "-5 dB", "0": "0 dB", "5": "5 dB"}
ablation_dataset = "ablation"

# 输出图片保存路径
output_dir = project_root / "figures" / "spectrograms"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {output_dir}")

# 语谱图参数
n_fft = 2048
hop_length = 512
win_length = 2048
window = "hann"

@dataclass
class RunConfig:
    model_set: str = "table1"
    datasets: list = field(default_factory=lambda: ["VCTK"])
    snrs: dict = field(default_factory=lambda: snr_mapping)
    duration: float = 3.0
    max_freq: int = 8000
    save: bool = True
    show: bool = True

def check_environment():
    try:
        import librosa  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except Exception as e:
        print(f"依赖检查失败: {e}")
        raise

def get_model_config(model_set: str):
    if model_set not in MODEL_SETS:
        print(f"未知模型集合: {model_set}，回退到 table1")
        model_set = "table1"
    return MODEL_SETS[model_set]

def find_audio_files(dataset, snr_folder, model_list):
    """
    查找指定数据集和信噪比下的所有音频文件
    """
    base_path = project_root / "samples" / dataset

    if not base_path.exists():
        print(f"警告: 路径不存在 {base_path}")
        return {}

    result = {}

    if snr_folder is None:
        search_path = base_path
    else:
        search_path = base_path / snr_folder

    if not search_path.exists():
        print(f"警告: 路径不存在 {search_path}")
        return {}

    audio_files = list(search_path.glob("*.wav"))
    for file_path in audio_files:
        model_name = file_path.stem
        if model_name in model_list:
            result[model_name] = {"path": str(file_path), "snr": snr_folder, "exists": True}

    return result

def plot_spectrograms_for_condition(dataset, snr_folder, run_cfg: RunConfig):
    """
    为特定数据集和信噪比条件绘制语谱图
    """
    if snr_folder is None:
        print(f"\n处理: {dataset}")
    else:
        print(f"\n处理: {dataset} - {snr_display.get(snr_folder, snr_folder)}")

    model_cfg = get_model_config(run_cfg.model_set)
    current_model_order = model_cfg["order"].get(dataset, [])
    current_display_names = model_cfg["display_names"].get(dataset, {})

    if not current_model_order:
        print(f"{dataset} 在模型集合 {run_cfg.model_set} 中未配置，跳过")
        return None

    files_dict = find_audio_files(dataset, snr_folder, current_model_order)
    if not files_dict:
        print(f"没有找到 {dataset} {snr_folder} 的音频文件")
        return None

    valid_files, valid_models, valid_display_names = [], [], []
    print("找到的模型:")
    for model in current_model_order:
        if model in files_dict:
            valid_files.append(files_dict[model]["path"])
            valid_models.append(model)
            valid_display_names.append(current_display_names.get(model, model))
            print(f"  ✓ {current_display_names.get(model, model)}")
        else:
            print(f"  ✗ {current_display_names.get(model, model)} (缺失)")

    if not valid_files:
        print("没有找到有效的音频文件")
        return None

    n_models = len(valid_files)
    if n_models <= 3:
        n_cols, n_rows = n_models, 1
    elif n_models <= 6:
        n_cols, n_rows = 3, (n_models + 2) // 3
    else:
        n_cols, n_rows = 4, (n_models + 3) // 4

    print(f"布局: {n_rows}行 x {n_cols}列")

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    axes = [plt.subplot(n_rows, n_cols, i + 1) for i in range(n_rows * n_cols)]

    if snr_folder is None:
        title = "Ablation"
    else:
        title = f"{dataset}  {snr_display.get(snr_folder, snr_folder)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for idx, (audio_file, model_name, display_name) in enumerate(zip(valid_files, valid_models, valid_display_names)):
        ax = axes[idx]
        try:
            if not os.path.exists(audio_file):
                ax.text(0.5, 0.5, f"File not found:\n{display_name}", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(display_name, fontsize=11, fontweight="bold")
                continue

            y, sr = librosa.load(audio_file, sr=None, duration=run_cfg.duration)
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
            DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            img = librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="viridis")
            ax.set_ylim(0, run_cfg.max_freq)

            ax.set_title(display_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Frequency (Hz)", fontsize=8)
            ax.tick_params(labelsize=7)

            if display_name == "Proposed_light" or model_name == "a_YLMS_IVA_SN" or model_name == "Proposed_base":
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {display_name}\n{str(e)[:30]}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(display_name, fontsize=11, fontweight="bold")

    for idx in range(len(valid_files), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)

    if "img" in locals():
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(img, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Intensity (dB)", fontsize=9)

    if run_cfg.save:
        if snr_folder is None:
            filename = f"{dataset}_{run_cfg.model_set}_spectrograms.png"
        else:
            snr_display_str = snr_folder.replace("N5", "-5")
            filename = f"{dataset}_{snr_display_str}dB_{run_cfg.model_set}_spectrograms.png"
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"已保存: {save_path}")

    if run_cfg.show:
        plt.show(block=False)
        plt.pause(0.5)

    return fig

def generate_all_spectrograms(run_cfg: RunConfig):
    """
    为所有数据集和信噪比生成语谱图
    """
    print("\n" + "=" * 60)
    print("开始生成语谱图...")
    print("=" * 60)

    figures = []

    if run_cfg.model_set == "table2":
        dataset = ablation_dataset
        fig = plot_spectrograms_for_condition(dataset, None, run_cfg)
        if fig:
            figures.append(fig)
    else:
        for dataset in run_cfg.datasets:
            snr_folders = run_cfg.snrs.get(dataset, [])
            for snr_folder in snr_folders:
                fig = plot_spectrograms_for_condition(dataset, snr_folder, run_cfg)
                if fig:
                    figures.append(fig)

    print("\n" + "=" * 60)
    print("所有语谱图已生成完毕！")
    print(f"保存位置: {output_dir}")
    print("=" * 60)

    print("\n生成的文件:")
    for png_file in output_dir.glob("*.png"):
        print(f"  - {png_file.name}")

    return figures

def parse_args():
    parser = argparse.ArgumentParser(description="MAB-Net 语谱图生成工具")
    parser.add_argument("--set", choices=MODEL_SETS.keys(), default="table1", help="模型集合：table1/table2")
    parser.add_argument("--datasets", nargs="*", default=None, help="限定数据集，如: VCTK")
    parser.add_argument("--snrs", nargs="*", default=None, help="限定SNR，如: N5 0 5")
    parser.add_argument("--duration", type=float, default=3.0, help="截取时长（秒）")
    parser.add_argument("--max-freq", type=int, default=8000, help="最高频率显示范围")
    parser.add_argument("--no-save", action="store_true", help="不保存图片")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口")
    return parser.parse_args()

# --- 主程序入口 ---
if __name__ == "__main__":
    print("=" * 60)
    print("MAB-Net 语谱图生成工具")
    print("=" * 60)

    args = parse_args()
    cfg = RunConfig(
        model_set=args.set,
        datasets=args.datasets if args.datasets else datasets,
        duration=args.duration,
        max_freq=args.max_freq,
        save=not args.no_save,
        show=not args.no_show,
    )

    # 限定SNR
    if args.snrs:
        cfg.snrs = {d: args.snrs for d in cfg.datasets}

    check_environment()
    generate_all_spectrograms(cfg)

    print("\n按回车键退出...")
    input()