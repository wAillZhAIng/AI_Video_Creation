import { useEffect, useMemo, useState } from 'react';
import { request } from '../lib/api';
import type { StudioShot } from '../lib/studio-contract';
import { FormField } from './ui';

export type CustomLoraItem = { name: string; strength: number };
export type CustomLoraSelection = { enabled: boolean; items: CustomLoraItem[] };

type Compatibility = { status: 'compatible' | 'partial' | 'incompatible' | 'unknown'; reason: string; base_model?: string | null };
type ScannedLora = { name: string; size_bytes: number; available_in_comfyui: boolean | null; compatibility: Compatibility };
type Limits = { max_loras: number; min_strength: number; max_strength: number; default_strength: number };
type ScanResult = { lora_dir: string; base_model: string | null; items: ScannedLora[]; limits: Limits };
type SavedConfig = { project_id: string; revision: number; project_default: CustomLoraSelection; shot_overrides: Record<string, CustomLoraSelection>; limits: Limits };

const STATUS_LABELS: Record<Compatibility['status'], string> = {
  compatible: '基座兼容', partial: '部分兼容', incompatible: '不兼容', unknown: '未校验',
};
const OFF: CustomLoraSelection = { enabled: false, items: [] };
const FALLBACK_LIMITS: Limits = { max_loras: 4, min_strength: 0.05, max_strength: 2, default_strength: 0.7 };

function clone(selection: CustomLoraSelection): CustomLoraSelection {
  return { enabled: Boolean(selection.enabled), items: (selection.items || []).map(item => ({ ...item })) };
}

function summary(selection: CustomLoraSelection) {
  if (!selection.enabled) return '关闭';
  if (!selection.items.length) return '已开启但未选择 LoRA';
  return selection.items.map(item => `${item.name.replace(/\.safetensors$/, '')} ×${item.strength}`).join('，');
}

export function ProjectCustomLoras({ projectId, shots, disabled }: { projectId: string; shots: StudioShot[]; disabled: boolean }) {
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [saved, setSaved] = useState<SavedConfig | null>(null);
  const [projectDefault, setProjectDefault] = useState<CustomLoraSelection>(OFF);
  const [overrides, setOverrides] = useState<Record<string, CustomLoraSelection>>({});
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [reload, setReload] = useState(0);
  const limits = scan?.limits || saved?.limits || FALLBACK_LIMITS;
  const usable = useMemo(() => (scan?.items || []).filter(item => item.compatibility.status !== 'incompatible'), [scan]);
  const dirty = saved !== null && JSON.stringify({ d: saved.project_default, o: saved.shot_overrides }) !== JSON.stringify({ d: projectDefault, o: overrides });

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError('');
    void Promise.all([
      request<ScanResult>('/api/providers/h3/loras', { signal: controller.signal }),
      request<SavedConfig>(`/api/studio/projects/${encodeURIComponent(projectId)}/custom-loras`, { signal: controller.signal }),
    ]).then(([scanned, config]) => {
      if (controller.signal.aborted) return;
      setScan(scanned); setSaved(config);
      setProjectDefault(clone(config.project_default || OFF));
      setOverrides(Object.fromEntries(Object.entries(config.shot_overrides || {}).map(([id, value]) => [id, clone(value)])));
    }).catch(cause => { if (!controller.signal.aborted) setError(String(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [projectId, reload]);

  async function save() {
    if (!saved || busy) return;
    setBusy(true); setError('');
    try {
      const result = await request<SavedConfig>(`/api/studio/projects/${encodeURIComponent(projectId)}/custom-loras`, {
        method: 'PUT',
        body: JSON.stringify({ expected_revision: saved.revision, project_default: projectDefault, shot_overrides: overrides }),
      });
      setSaved(result);
      setProjectDefault(clone(result.project_default));
      setOverrides(Object.fromEntries(Object.entries(result.shot_overrides || {}).map(([id, value]) => [id, clone(value)])));
      setNotice('LoRA 设置已保存。之后新提交的分镜任务会冻结实际 LoRA 名称与强度；已提交任务与重试保持原快照。');
    } catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }

  const locked = disabled || busy || loading;
  return <section className="project-reference-videos project-custom-loras" aria-label="自定义 LoRA">
    <div className="studio-section-heading"><div><h3>自定义 LoRA（可选）</h3>
      <p>自动扫描 <code>{scan?.lora_dir || 'ComfyUI/models/loras'}</code>，新放入的 <code>.safetensors</code> 点“重新扫描”即可使用。开启后在 H3 模型链（加速 LoRA 之后）动态串联；关闭时工作流完全不变。</p></div>
      <button type="button" className="button button-secondary button-small" disabled={locked} onClick={() => setReload(value => value + 1)}>重新扫描</button></div>
    {scan ? <p className="form-hint">当前基座：<code>{scan.base_model || '未识别'}</code> · 可用 {usable.length} / {scan.items.length} 个（不兼容的 LoRA 不可选择）</p> : null}
    {loading ? <p role="status">扫描 LoRA 目录并校验基座兼容性…</p> : null}
    {error ? <p role="alert" className="inline-notice warning">{error}</p> : null}
    {notice ? <p role="status" className="inline-notice">{notice}</p> : null}
    {scan?.items.length ? <ul className="custom-lora-scan">{scan.items.map(item => <li key={item.name}>
      <strong>{item.name}</strong> · {(item.size_bytes / 1048576).toFixed(0)} MB · <span className={`custom-lora-status custom-lora-${item.compatibility.status}`}>{STATUS_LABELS[item.compatibility.status]}</span>
      {item.available_in_comfyui === false ? ' · ComfyUI 尚未识别（需刷新 ComfyUI 模型列表）' : ''}<br/><small>{item.compatibility.reason}</small></li>)}</ul>
      : (!loading ? <p className="form-hint">LoRA 目录中没有可选的自定义 LoRA。</p> : null)}
    <LoraSelectionEditor title="项目默认" selection={projectDefault} options={usable} limits={limits} disabled={locked} onChange={setProjectDefault}/>
    <section className="project-reference-shot-list">
      <h4>逐镜开关</h4>
      <p className="form-hint">未设置覆盖时继承项目默认；可为单个分镜单独关闭、开启或调整强度（成片审核回退重绘时同样生效）。</p>
      {shots.map(shot => {
        const explicit = Object.prototype.hasOwnProperty.call(overrides, shot.id);
        const current = overrides[shot.id];
        return <details className="workspace-details project-reference-shot" key={shot.id}>
          <summary>第 {shot.order} 镜 · {shot.title || shot.scene_name || shot.id} · {explicit ? `本镜覆盖：${summary(current)}` : `继承项目默认：${summary(projectDefault)}`}</summary>
          <div className="asset-actions">
            <button type="button" className="button button-secondary button-small" disabled={locked} onClick={() => setOverrides(previous => ({ ...previous, [shot.id]: explicit ? previous[shot.id] : clone({ ...projectDefault, enabled: true }) }))}>创建／编辑本镜覆盖</button>
            {explicit ? <button type="button" className="button button-secondary button-small" disabled={locked} onClick={() => setOverrides(previous => { const next = { ...previous }; delete next[shot.id]; return next; })}>恢复项目默认</button> : null}
            <button type="button" className="button button-ghost-danger button-small" disabled={locked} onClick={() => setOverrides(previous => ({ ...previous, [shot.id]: { enabled: false, items: [] } }))}>本镜不使用 LoRA</button>
          </div>
          {explicit ? <LoraSelectionEditor title="本镜覆盖" selection={current} options={usable} limits={limits} disabled={locked}
            onChange={value => setOverrides(previous => ({ ...previous, [shot.id]: value }))}/> : null}
        </details>;
      })}
    </section>
    <div className="studio-actions"><button type="button" className="button button-primary" disabled={locked || !dirty} onClick={() => void save()}>保存 LoRA 设置</button>
      {dirty && saved ? <button type="button" className="button button-secondary" disabled={locked} onClick={() => { setProjectDefault(clone(saved.project_default)); setOverrides(Object.fromEntries(Object.entries(saved.shot_overrides || {}).map(([id, value]) => [id, clone(value)]))); }}>放弃未保存选择</button> : null}</div>
  </section>;
}

function LoraSelectionEditor({ title, selection, options, limits, disabled, onChange }: {
  title: string; selection: CustomLoraSelection; options: ScannedLora[]; limits: Limits; disabled: boolean;
  onChange: (value: CustomLoraSelection) => void;
}) {
  const [choice, setChoice] = useState('');
  const remaining = options.filter(option => !selection.items.some(item => item.name === option.name));
  return <fieldset className="project-reference-binding-scope"><legend>{title}</legend>
    <label className="studio-check"><input type="checkbox" checked={selection.enabled} disabled={disabled}
      onChange={event => onChange({ ...selection, enabled: event.target.checked })}/>启用附加 LoRA</label>
    {selection.items.map((item, index) => <div className="toolbar" key={item.name}>
      <strong>{item.name}</strong>
      <FormField label={`强度 ${item.strength}`}><input type="range" min={limits.min_strength} max={limits.max_strength} step="0.05" value={item.strength} disabled={disabled || !selection.enabled}
        onChange={event => onChange({ ...selection, items: selection.items.map((entry, position) => position === index ? { ...entry, strength: Number(event.target.value) } : entry) })}/></FormField>
      <button type="button" className="button button-ghost-danger button-small" disabled={disabled}
        onClick={() => onChange({ ...selection, items: selection.items.filter((_, position) => position !== index) })}>移除</button>
    </div>)}
    <div className="toolbar"><FormField label="添加 LoRA"><select value={choice} disabled={disabled || selection.items.length >= limits.max_loras} onChange={event => setChoice(event.target.value)}>
      <option value="">请选择</option>{remaining.map(option => <option key={option.name} value={option.name}>{option.name}（{STATUS_LABELS[option.compatibility.status]}）</option>)}</select></FormField>
      <button type="button" className="button button-secondary button-small" disabled={disabled || !choice || selection.items.length >= limits.max_loras}
        onClick={() => { onChange({ enabled: true, items: [...selection.items, { name: choice, strength: limits.default_strength }] }); setChoice(''); }}>添加</button></div>
  </fieldset>;
}
