"""Check preservation of the old 1k results and write the completed 5k comparison."""
import argparse
from pathlib import Path

import numpy as np

from mms_eval.artifacts import load_evaluation
from mms_eval.utils import read_json, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    completion = read_json(out/'completion.json')
    assert completion['status'] == 'complete' and completion['common_n'] == 5000
    assert sha256_file(out/'expansion_request.json') == completion['request_sha256']
    assert sha256_file(out/'ddim50_5000.jsonl') == completion['merged_manifest_sha256']
    old_root = root/'sampler_comparison_1000_v5_20260913/DDIM50'
    old_report, old_rows, _, _ = load_evaluation(old_root)
    new_root = out/'evaluation_DDIM50_A'
    new_report, new_rows, _, _ = load_evaluation(new_root)
    assert len(old_rows) == 1000 and len(new_rows) == 5000
    assert new_rows[:1000] == old_rows, 'Preserved images must retain their per-image A scores'
    assert old_report['features_sha256'] == sha256_file(old_root/'features.npz')
    assert new_report['features_sha256'] == sha256_file(new_root/'features.npz')
    feature_checks = {}
    with np.load(old_root/'features.npz', allow_pickle=False) as before, \
            np.load(new_root/'features.npz', allow_pickle=False) as after:
        assert set(before.files) == set(after.files)
        for name in before.files:
            feature_checks[name] = bool(np.array_equal(before[name], after[name][:1000]))
    assert all(feature_checks.values()), feature_checks
    write_json(out/'prefix_reproduction.json', {
        'original_n': 1000, 'expanded_n': 5000, 'original_A_score_rows_exactly_preserved': True,
        'feature_arrays_exactly_preserved': feature_checks,
        'original_report_sha256': sha256_file(old_root/'report.json'),
        'expanded_report_sha256': sha256_file(new_root/'report.json'),
    })
    reports = {}
    for evaluator in ['A', 'B']:
        for sampler, path in [('DDPM250', root/f'evaluation_{evaluator}_delivery_5000_v5'),
                              ('DDIM50', out/f'evaluation_DDIM50_{evaluator}')]:
            report, _, _, _ = load_evaluation(path)
            expected = next(r for r in completion['comparisons'][evaluator] if r['name'] == sampler)
            assert sha256_file(path/'report.json') == expected['report_sha256']
            reports[evaluator, sampler] = report
    differences = {
        e: reports[e, 'DDIM50']['mms']['value']-reports[e, 'DDPM250']['mms']['value'] for e in ['A', 'B']}
    overlap = completion['DDIM50_candidate_overlap']
    rows = ['# DDIM50扩至5,000张：A/B同图数比较', '',
            '日期：2026-09-13。保留原有1,000张DDIM50，固定新增种子1000–4999，共补4,000张；生成完成后使用A/B两位评估器，与原有DDPM250的固定5,000张分别比较。', '',
            f"自动候选率差（DDIM50减DDPM250）：A为{differences['A']*100:+.2f}个百分点，B为{differences['B']*100:+.2f}个百分点。正值表示这位评估器在DDIM50图池中筛出了更多候选；没有人工标签时不能称为真实混淆率差。", '',
            '## 自动候选结果', '',
            '| 评估器 | 采样设置 | 图数 | 候选数 | MMS候选率 | 95% Wilson区间 | 平均熵 |',
            '|---|---|---:|---:|---:|---|---:|']
    for evaluator in ['A', 'B']:
        for sampler in ['DDPM250', 'DDIM50']:
            m = reports[evaluator, sampler]['mms']
            rows.append(f"| {evaluator} | {sampler} | 5,000 | {m['candidate_count']} | {m['value']:.2%} | {m['ci95'][0]:.2%}–{m['ci95'][1]:.2%} | {m['mean_entropy']:.6f} |")
    rows += ['', '区间以固定评估器、固定真实校准和独立生成图为条件；不包含人工语义误差、重新训练或参考选择的不确定性。', '',
             '## 质量指标', '',
             'FID/KID、IS和生成P/R使用独立的固定质量后端。下表列A评测路径的结果，B路径完整数值保存在比较文件中。', '',
             '| 采样设置 | FID ↓ | KID ↓ | IS | 生成P | 生成R |',
             '|---|---:|---:|---:|---:|---:|']
    for sampler in ['DDPM250', 'DDIM50']:
        d = reports['A', sampler]['distribution']
        rows.append(f"| {sampler} | {d['fid']['value']:.5f} | {d['kid']['mean']:.8f} | {d['is']['mean']:.5f} | {d['precision_recall']['precision']:.4f} | {d['precision_recall']['recall']:.4f} |")
    rows += ['',
             '同一14,626张质量真实池；Clean-FID clean FID/KID、fidelity IS、官方VGG PR。两边PR各5,000张、k=3；KID每边1,000张子集重复100次，IS固定10个分块。子集标准差不作为方法差异置信区间。', '',
             '[A比较图](comparison_A/comparison.pdf) · [B比较图](comparison_B/comparison.pdf) · [DDIM的A/B比较](comparison_DDIM50_AB/comparison.pdf)', '',
             '## 两位评估器的一致性', '',
             f"在相同5,000张DDIM图片上，两者都标为候选的有{overlap['both']}张，仅A标出的有{overlap['A_only']}张，仅B标出的有{overlap['B_only']}张，两者均未标出的有{overlap['neither']}张。", '',
             '候选重叠描述评估器的一致性，不是人工准确率。A/B各用自己的真实校准阈值，不混合为一个候选率。', '',
             '## 扩样与复现核验', '',
             '- 原有种子0–999的图片未被替换，新增种子1000–4999按两路各2,000张生成。每批4张，固定500k权重、DDIM50、eta=0和原有预处理/量化规则。',
             '- 5,000个种子完整无重复，图像文件内容哈希各不相同，与固定DDPM250的5k图池无相同文件内容。两路的科学采样合同一致，分片起始种子单独记录。',
             '- 原有1,000张在扩至5k之后的A逐图分数、语义特征和全部质量特征均逐元素保持一致；不会因为合并或缓存改变已有结果。',
             '- 两个采样设置按每位评估器分别通过同N、同真实参考、同阈值与同质量后端检查。旧DDPM250结果复用其已验收报告，未重新抽图。',
             '- 生成、缓存和评测均保留分阶段记录；生成器优化及全量B的50,010张实验未启动。', '',
             '[扩样输入核验](cohort_verification.json) · [原1k精确复现](prefix_reproduction.json) · [完成证明](completion.json) · [冻结采样计划](expansion_request.json)', '',
             '## 解释边界', '',
             '这是现有500k权重路径对应的两种采样设置和两个固定图池。历史DDPM的逐图噪声种子未建立，两组不是配对样本，生成时期和执行细节也不同。DDIM现存权重已核对哈希；历史DDPM记录中的相同路径不能独立认证其当时权重字节。不能把全部指标差异归因于采样器，也不能据此宣称跨模型家族或人工语义效度已完成。', '',
             '1k与5k比较涉及样本量变化，FID/IS/PR可能随N变化；正式采样设置比较以本页的同5k表为准。']
    (out/'RESULTS.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    print('SAMPLER_5000_SUMMARY_AND_PREFIX_VERIFICATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
