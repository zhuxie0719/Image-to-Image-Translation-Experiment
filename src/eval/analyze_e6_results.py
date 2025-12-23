"""
分析 E6 (Pix2Pix + Feature Matching) 实验结果
并与 E2 (Pix2Pix L1+GAN) 进行对比
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_e6_results():
    """分析 E6 实验结果"""
    print("=" * 70)
    print("实验 E6 (Pix2Pix + Feature Matching) 结果分析")
    print("=" * 70)
    
    # E6 结果
    e6_results = {
        "最佳 epoch": 31,
        "最佳 Val L1/MAE": 0.1119,
        "最佳 PSNR": 16.132,
        "最佳 SSIM": 0.441,
        "最后 epoch": 50,
        "最后 Val L1/MAE": 0.1224,
        "最后 PSNR": 15.658,
        "最后 SSIM": 0.396,
    }
    
    # E2 结果（从实验日志）
    e2_results = {
        "最佳 epoch": 36,
        "最佳 Val L1/MAE": 0.1109,
        "最佳 PSNR": 16.275,
        "最佳 SSIM": 0.435,
        "最后 epoch": 50,
        "最后 Val L1/MAE": 0.1122,
        "最后 PSNR": 16.150,
        "最后 SSIM": 0.449,
    }
    
    print("\n[E6] E6 (L1 + GAN + Feature Matching) 结果:")
    print(f"   最佳 epoch: {e6_results['最佳 epoch']}")
    print(f"   - Val L1/MAE: {e6_results['最佳 Val L1/MAE']:.4f} ↓")
    print(f"   - PSNR: {e6_results['最佳 PSNR']:.3f} dB ↑")
    print(f"   - SSIM: {e6_results['最佳 SSIM']:.3f} ↑")
    print(f"\n   最后 epoch ({e6_results['最后 epoch']}):")
    print(f"   - Val L1/MAE: {e6_results['最后 Val L1/MAE']:.4f} ↑ (比最佳差)")
    print(f"   - PSNR: {e6_results['最后 PSNR']:.3f} dB ↓ (比最佳差)")
    print(f"   - SSIM: {e6_results['最后 SSIM']:.3f} ↓ (比最佳差)")
    
    print("\n" + "-" * 70)
    print("[E2] E2 (L1 + GAN) 结果（对比基准）:")
    print(f"   最佳 epoch: {e2_results['最佳 epoch']}")
    print(f"   - Val L1/MAE: {e2_results['最佳 Val L1/MAE']:.4f} ↓")
    print(f"   - PSNR: {e2_results['最佳 PSNR']:.3f} dB ↑")
    print(f"   - SSIM: {e2_results['最佳 SSIM']:.3f} ↑")
    print(f"\n   最后 epoch ({e2_results['最后 epoch']}):")
    print(f"   - Val L1/MAE: {e2_results['最后 Val L1/MAE']:.4f}")
    print(f"   - PSNR: {e2_results['最后 PSNR']:.3f} dB")
    print(f"   - SSIM: {e2_results['最后 SSIM']:.3f}")
    
    print("\n" + "=" * 70)
    print("[对比] E2 vs E6 对比分析（最佳 epoch）")
    print("=" * 70)
    
    # 计算差异
    l1_diff = e6_results['最佳 Val L1/MAE'] - e2_results['最佳 Val L1/MAE']
    psnr_diff = e6_results['最佳 PSNR'] - e2_results['最佳 PSNR']
    ssim_diff = e6_results['最佳 SSIM'] - e2_results['最佳 SSIM']
    
    print(f"\n[L1/MAE] Val L1/MAE: E6 比 E2 {'高' if l1_diff > 0 else '低'} {abs(l1_diff):.4f}")
    print(f"   E2: {e2_results['最佳 Val L1/MAE']:.4f} | E6: {e6_results['最佳 Val L1/MAE']:.4f}")
    if l1_diff > 0:
        print("   [WARN] E6 的像素误差略高于 E2（Feature Matching 可能略微增加了重建难度）")
    else:
        print("   [OK] E6 的像素误差更低（Feature Matching 有助于像素级重建）")
    
    print(f"\n[PSNR] PSNR: E6 比 E2 {'高' if psnr_diff > 0 else '低'} {abs(psnr_diff):.3f} dB")
    print(f"   E2: {e2_results['最佳 PSNR']:.3f} dB | E6: {e6_results['最佳 PSNR']:.3f} dB")
    if psnr_diff < 0:
        print("   [WARN] E6 的 PSNR 略低于 E2（约 -0.14 dB，差异较小）")
    else:
        print("   [OK] E6 的 PSNR 更高")
    
    print(f"\n[SSIM] SSIM: E6 比 E2 {'高' if ssim_diff > 0 else '低'} {abs(ssim_diff):.3f}")
    print(f"   E2: {e2_results['最佳 SSIM']:.3f} | E6: {e6_results['最佳 SSIM']:.3f}")
    if ssim_diff > 0:
        print("   [OK] E6 的 SSIM 更高（Feature Matching 有助于结构相似度）")
    else:
        print("   [WARN] E6 的 SSIM 略低")
    
    print("\n" + "=" * 70)
    print("[关键发现] 关键发现")
    print("=" * 70)
    
    print("\n1. **训练稳定性**")
    print("   - E6 的最佳 epoch 在 31，E2 在 36")
    print("   - E6 在 31 epoch 后性能开始下降，可能存在轻微过拟合")
    print("   - E2 在 36 epoch 后仍保持稳定，最后 epoch 的 SSIM 甚至略有提升")
    
    print("\n2. **Feature Matching 的影响**")
    print("   - E6 的最佳 SSIM (0.441) 略高于 E2 (0.435)，说明 Feature Matching")
    print("     有助于提升结构相似度和感知质量")
    print("   - 但 E6 的 PSNR 和 L1/MAE 略差，说明在像素级精度上略有牺牲")
    print("   - 这可能是因为 Feature Matching 更关注特征空间的一致性，")
    print("     而非严格的像素级匹配")
    
    print("\n3. **后期训练行为**")
    print("   - E6 在 50 epoch 时所有指标都低于最佳 epoch，显示明显的过拟合")
    print("   - E2 在 50 epoch 时 L1 略有上升，但 SSIM 提升，说明模型在")
    print("     优化感知质量而非像素精度")
    print("   - 建议：E6 应该在 31 epoch 附近早停，或降低学习率")
    
    print("\n4. **实验配置**")
    print("   - E6 使用 λ_FM = 10.0，可能偏大，导致 Feature Matching 损失")
    print("     过度影响训练，建议尝试 λ_FM = 1.0 或 5.0")
    print("   - 两个实验都使用 strong 增强，配置一致，对比公平")
    
    print("\n" + "=" * 70)
    print("[建议] 建议")
    print("=" * 70)
    
    print("\n1. **早停策略**：E6 应该在 epoch 31 附近停止训练")
    print("2. **超参数调整**：尝试降低 λ_FM 到 1.0-5.0，观察是否改善训练稳定性")
    print("3. **定性分析**：查看 epoch 31 和 epoch 50 的样例图，对比视觉质量差异")
    print("4. **与 E7 对比**：等待 E7 (Perceptual Loss) 结果，对比三种损失函数的组合效果")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_e6_results()

