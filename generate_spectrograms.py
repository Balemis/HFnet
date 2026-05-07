#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# 打印调试信息
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"Matplotlib 版本: {matplotlib.__version__}")
except ImportError as e:
    print(f"导入 matplotlib 失败: {e}")
    print("请运行: python -m pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError as e:
    print(f"导入 numpy 失败: {e}")
    sys.exit(1)

try:
    import librosa
    import librosa.display
    print(f"Librosa 版本: {librosa.__version__}")
except ImportError as e:
    print(f"导入 librosa 失败: {e}")
    print("请运行: python -m pip install librosa")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore')

# --- 配置参数 ---
project_root = Path.cwd()
print(f"项目根路径: {project_root}")

# 模型名称映射（按数据集区分）
model_display_names = {
    'CHIME-3': {
        'noisy': 'Noisy',
        'GTCRN': 'GTCRN',
        'FasNet': 'FasNet',
        'FasNet-TAC': 'FasNet-TAC',
        'proposed': 'Proposed (MAB-Net)'
    },
    'VCTK': {
        'clean': 'Clean',
        'noisy': 'Noisy',
        'GTCRN': 'GTCRN',
        'FasNet': 'FasNet',
        'FasNet-TAC': 'FasNet-TAC',
        'proposed': 'Proposed (MAB-Net)'
    },
    'wsj0': {
        'clean': 'Clean',
        'noisy': 'Noisy',
        'GTCRN': 'GTCRN',
        'FasNet': 'FasNet',
        'FasNet-TAC': 'FasNet-TAC',
        'proposed': 'Proposed (MAB-Net)'
    }
}

# 模型顺序（按数据集区分）
model_order = {
    'CHIME-3': ['noisy', 'GTCRN', 'FasNet', 'FasNet-TAC', 'proposed'],  # CHIME-3没有clean
    'VCTK': ['clean', 'noisy', 'GTCRN', 'FasNet', 'FasNet-TAC', 'proposed'],
    'wsj0': ['clean', 'noisy', 'GTCRN', 'FasNet', 'FasNet-TAC', 'proposed']
}

# 数据集和信噪比配置
datasets = ['CHIME-3', 'VCTK', 'wsj0']
snr_mapping = {
    'CHIME-3': ['N5'],  # CHIME-3只有N5
    'VCTK': ['N5', '0', '5'],
    'wsj0': ['N5', '0', '5']
}
snr_display = {
    'N5': '-5 dB',
    '0': '0 dB',
    '5': '5 dB'
}

# 输出图片保存路径
output_dir = project_root / 'figures' / 'spectrograms'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"输出目录: {output_dir}")

# 语谱图参数
n_fft = 2048
hop_length = 512
win_length = 2048
window = 'hann'

def find_audio_files(dataset, snr_folder):
    """
    查找指定数据集和信噪比下的所有音频文件
    """
    base_path = project_root / 'samples' / dataset
    
    if not base_path.exists():
        print(f"警告: 路径不存在 {base_path}")
        return {}
    
    result = {}
    
    if dataset == 'CHIME-3':
        # CHIME-3：文件直接在dataset目录下
        audio_files = list(base_path.glob("*.wav"))
        for file_path in audio_files:
            model_name = file_path.stem
            # CHIME-3只有这些模型
            if model_name in ['noisy', 'GTCRN', 'FasNet', 'FasNet-TAC', 'proposed']:
                result[model_name] = {
                    'path': str(file_path),
                    'snr': 'N5',
                    'exists': True
                }
    else:
        # VCTK和wsj0：按信噪比文件夹组织
        search_path = base_path / snr_folder
        if not search_path.exists():
            print(f"警告: 路径不存在 {search_path}")
            return {}
        
        audio_files = list(search_path.glob("*.wav"))
        for file_path in audio_files:
            model_name = file_path.stem
            # VCTK和wsj0包含所有模型
            if model_name in ['clean', 'noisy', 'GTCRN', 'FasNet', 'FasNet-TAC', 'proposed']:
                result[model_name] = {
                    'path': str(file_path),
                    'snr': snr_folder,
                    'exists': True
                }
    
    return result

def plot_spectrograms_for_condition(dataset, snr_folder, save=True):
    """
    为特定数据集和信噪比条件绘制语谱图
    """
    print(f"\n处理: {dataset} - {snr_display.get(snr_folder, snr_folder)}")
    
    # 查找音频文件
    files_dict = find_audio_files(dataset, snr_folder)
    
    if not files_dict:
        print(f"没有找到 {dataset} {snr_folder} 的音频文件")
        return None
    
    # 获取该数据集的模型顺序和显示名称
    current_model_order = model_order[dataset]
    current_display_names = model_display_names[dataset]
    
    # 准备数据
    valid_files = []
    valid_models = []
    valid_display_names = []
    
    print("找到的模型:")
    for model in current_model_order:
        if model in files_dict:
            valid_files.append(files_dict[model]['path'])
            valid_models.append(model)
            valid_display_names.append(current_display_names.get(model, model))
            print(f"  ✓ {current_display_names.get(model, model)}")
        else:
            print(f"  ✗ {current_display_names.get(model, model)} (缺失)")
    
    if not valid_files:
        print(f"没有找到有效的音频文件")
        return None
    
    # 计算子图布局
    n_models = len(valid_files)
    
    # 根据模型数量决定布局
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = (n_models + 2) // 3
    else:
        n_cols = 4
        n_rows = (n_models + 3) // 4
    
    print(f"布局: {n_rows}行 x {n_cols}列")
    
    # 创建图形
    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = [plt.subplot(1, 1, 1)]
    else:
        axes = []
        for i in range(n_rows * n_cols):
            axes.append(plt.subplot(n_rows, n_cols, i + 1))
    
    # 标题 - 根据数据集不同设置不同的标题格式
    if dataset == 'CHIME-3':
        # CHIME-3: 只显示数据集名称和Blind test
        title = f'{dataset} (Blind test)'
    else:
        # VCTK和wsj0: 显示数据集名称和信噪比
        title = f'{dataset}  {snr_display.get(snr_folder, snr_folder)}'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # 为每个模型绘制语谱图
    for idx, (audio_file, model_name, display_name) in enumerate(zip(valid_files, valid_models, valid_display_names)):
        ax = axes[idx]
        
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_file):
                ax.text(0.5, 0.5, f'File not found:\n{display_name}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(display_name, fontsize=11, fontweight='bold')
                continue
            
            # 加载音频
            y, sr = librosa.load(audio_file, sr=None, duration=3.0)
            
            # 计算STFT
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, 
                            win_length=win_length, window=window)
            DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # 绘制语谱图
            img = librosa.display.specshow(DB, sr=sr, hop_length=hop_length,
                                          x_axis='time', y_axis='hz',
                                          ax=ax, cmap='viridis')
            
            # 设置频率范围
            ax.set_ylim(0, 8000)
            
            # 设置标题和标签
            ax.set_title(display_name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Frequency (Hz)', fontsize=8)
            ax.tick_params(labelsize=7)
            
            # 如果是proposed模型，添加红框
            if model_name == 'proposed':
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2.5)
                    
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {display_name}\n{str(e)[:30]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(display_name, fontsize=11, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(len(valid_files), len(axes)):
        axes[idx].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    # 添加颜色条
    if 'img' in locals():
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Intensity (dB)', fontsize=9)
    
    # 保存图片
    if save:
        if dataset == 'CHIME-3':
            # CHIME-3的文件名不加dB信息
            filename = f"{dataset}_spectrograms.png"
        else:
            snr_display_str = snr_folder.replace('N5', '-5')
            filename = f"{dataset}_{snr_display_str}dB_spectrograms.png"
        
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"已保存: {save_path}")
    
    plt.show(block=False)
    plt.pause(0.5)
    return fig

def check_environment():
    """
    检查运行环境
    """
    print("\n环境检查:")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查samples目录
    samples_path = project_root / 'samples'
    if samples_path.exists():
        print(f"samples目录存在: {samples_path}")
        for dataset in datasets:
            dataset_path = samples_path / dataset
            if dataset_path.exists():
                print(f"  {dataset}:")
                # 列出找到的wav文件
                wav_files = list(dataset_path.rglob("*.wav"))
                for wav in wav_files:
                    rel_path = wav.relative_to(dataset_path)
                    print(f"    - {rel_path}")
            else:
                print(f"  {dataset}: 目录不存在")
    else:
        print(f"samples目录不存在: {samples_path}")
    
    print()

def generate_all_spectrograms():
    """
    为所有数据集和信噪比生成语谱图
    """
    print("\n" + "=" * 60)
    print("开始生成语谱图...")
    print("=" * 60)
    
    figures = []
    
    for dataset in datasets:
        snr_folders = snr_mapping[dataset]
        
        for snr_folder in snr_folders:
            fig = plot_spectrograms_for_condition(dataset, snr_folder, save=True)
            if fig:
                figures.append(fig)
    
    print("\n" + "=" * 60)
    print(f"所有语谱图已生成完毕！")
    print(f"保存位置: {output_dir}")
    print("=" * 60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    for png_file in output_dir.glob("*.png"):
        print(f"  - {png_file.name}")
    
    return figures

# --- 主程序入口 ---
if __name__ == "__main__":
    print("=" * 60)
    print("MAB-Net 语谱图生成工具")
    print("=" * 60)
    
    # 检查环境
    check_environment()
    
    # 生成所有语谱图
    generate_all_spectrograms()
    
    print("\n按回车键退出...")
    input()