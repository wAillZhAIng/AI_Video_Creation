import { useState, type FormEvent } from 'react';
import { Film, Scissors } from 'lucide-react';
import { request } from '../lib/api';
import type { ReferenceVideoAsset } from '../lib/asset-library-contract';
import type { QualitySettings } from '../lib/quality-contract';
import { FormField } from './ui';

export type LongVideoSplitShot = {
  index?: number;
  shot_id?: string;
  order?: number;
  title?: string;
  trim_start_seconds: number;
  trim_end_seconds: number;
  duration_seconds: number;
  source_cut_strategy: string;
  vlm_described?: boolean;
};

export type LongVideoReplicationResult = {
  status: string;
  project_id: string;
  project_name: string;
  video_id: string;
  video_name: string;
  aspect_ratio: string;
  source_duration_seconds: number;
  detected_scene_cuts: number;
  shot_count: number;
  total_planned_duration_seconds: number;
  reference_role: string;
  include_audio: boolean;
  vlm_described_count: number;
  shots: LongVideoSplitShot[];
};

type SplitPreview = {
  video_id: string;
  video_name: string;
  source_duration_seconds: number;
  aspect_ratio: string;
  detected_scene_cuts: number;
  shot_count: number;
  total_planned_duration_seconds: number;
  shots: LongVideoSplitShot[];
};

const ROLE_OPTIONS = [
  { value: 'combined', label: '综合参考（动作 + 运镜 + 光影）' },
  { value: 'action', label: '仅动作与物理交互' },
  { value: 'camera', label: '仅运镜与景别节奏' },
  { value: 'style', label: '仅光影与画面质感' },
  { value: 'structure', label: '节奏结构与物理连贯性' },
];

export function LongVideoReplicationPanel({
  disabled = false,
  quality,
  initialVideo,
  defaultOpen = false,
  onBusyChange,
  onCreated,
}: {
  disabled?: boolean;
  quality?: QualitySettings;
  initialVideo?: { id: string; name: string; duration_seconds?: number };
  defaultOpen?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onCreated: (result: LongVideoReplicationResult) => Promise<void> | void;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [sourceMode, setSourceMode] = useState<'upload' | 'library'>(initialVideo ? 'library' : 'upload');
  const [file, setFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [libraryVideos, setLibraryVideos] = useState<ReferenceVideoAsset[]>([]);
  const [libraryLoaded, setLibraryLoaded] = useState(false);
  const [videoId, setVideoId] = useState(initialVideo?.id || '');
  const [projectName, setProjectName] = useState('');
  const [minShot, setMinShot] = useState(2.5);
  const [targetShot, setTargetShot] = useState(5.0);
  const [maxShot, setMaxShot] = useState(8.0);
  const [sceneThreshold, setSceneThreshold] = useState(0.25);
  const [referenceRole, setReferenceRole] = useState('combined');
  const [customPromptPrefix, setCustomPromptPrefix] = useState('');
  const [includeAudio, setIncludeAudio] = useState(false);
  const [analyzeWithVlm, setAnalyzeWithVlm] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<SplitPreview | null>(null);
  const [result, setResult] = useState<LongVideoReplicationResult | null>(null);

  async function loadLibrary() {
    setError('');
    try {
      const data = await request<{ items: ReferenceVideoAsset[] }>('/api/reference-videos');
      setLibraryVideos(data.items || []);
      setLibraryLoaded(true);
      if (!videoId && data.items?.length) {
        setVideoId(data.items[0].id);
      }
    } catch (e) {
      setError(`读取参考视频库失败：${String(e)}`);
    }
  }

  function handleToggle(nextOpen: boolean) {
    setOpen(nextOpen);
    if (nextOpen && sourceMode === 'library' && !libraryLoaded && !initialVideo) {
      void loadLibrary();
    }
  }

  async function handlePreview() {
    const activeVideoId = initialVideo?.id || videoId;
    if (!activeVideoId || busy || disabled) return;
    setBusy('正在按转场检测切分预览…');
    onBusyChange?.(true);
    setError('');
    try {
      const form = new FormData();
      form.set('video_id', activeVideoId);
      form.set('min_shot_seconds', String(minShot));
      form.set('target_shot_seconds', String(targetShot));
      form.set('max_shot_seconds', String(maxShot));
      form.set('scene_threshold', String(sceneThreshold));
      form.set('reference_role', referenceRole);
      const data = await request<SplitPreview>('/api/studio/preview-long-video-split', { method: 'POST', body: form });
      setPreview(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy('');
      onBusyChange?.(false);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (busy || disabled) return;
    const activeVideoId = initialVideo?.id || (sourceMode === 'library' ? videoId : '');
    if (sourceMode === 'upload' && !initialVideo && !file) {
      setError('请先选择一条本地参考视频（支持 3–5 分钟长视频）。');
      return;
    }
    if ((sourceMode === 'library' || initialVideo) && !activeVideoId) {
      setError('请先从参考视频库中选择一条视频。');
      return;
    }

    setBusy(analyzeWithVlm ? '正在切分镜头并调用本机视觉模型反推画面描述…' : '正在按转场切分长视频并创建多镜项目…');
    onBusyChange?.(true);
    setError('');
    setResult(null);
    try {
      const form = new FormData();
      if (initialVideo || sourceMode === 'library') {
        form.set('video_id', activeVideoId);
      } else if (file) {
        form.set('file', file);
      }
      if (projectName.trim()) form.set('project_name', projectName.trim());
      form.set('min_shot_seconds', String(minShot));
      form.set('target_shot_seconds', String(targetShot));
      form.set('max_shot_seconds', String(maxShot));
      form.set('scene_threshold', String(sceneThreshold));
      form.set('reference_role', referenceRole);
      form.set('include_audio', String(includeAudio));
      form.set('custom_prompt_prefix', customPromptPrefix);
      form.set('analyze_with_vlm', String(analyzeWithVlm));
      if (quality) form.set('quality_json', JSON.stringify(quality));

      const created = await request<LongVideoReplicationResult>('/api/studio/import-long-video', {
        method: 'POST',
        body: form,
      });
      setResult(created);
      setFile(null);
      setFileInputKey(v => v + 1);
      setPreview(null);
      await onCreated(created);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
      onBusyChange?.(false);
    }
  }

  const canSubmit = !busy && !disabled && (
    Boolean(initialVideo?.id) ||
    (sourceMode === 'upload' ? Boolean(file) : Boolean(videoId))
  );

  const body = (
    <form onSubmit={e => void handleSubmit(e)}>
      <p className="form-hint">
        自动按画面转场将 3–5 分钟长视频切分为多个 2.5–8 秒单镜（严格满足 H3 原生 15 秒上限并避开长镜物理畸变），自动为每一镜绑定独立参考片段与防畸变物理约束。
      </p>

      {!initialVideo ? (
        <div className="toolbar">
          <button
            type="button"
            className={`button button-small ${sourceMode === 'upload' ? 'button-primary' : 'button-secondary'}`}
            disabled={Boolean(busy) || disabled}
            onClick={() => setSourceMode('upload')}
          >
            上传新长视频
          </button>
          <button
            type="button"
            className={`button button-small ${sourceMode === 'library' ? 'button-primary' : 'button-secondary'}`}
            disabled={Boolean(busy) || disabled}
            onClick={() => {
              setSourceMode('library');
              if (!libraryLoaded) void loadLibrary();
            }}
          >
            从参考视频库选择
          </button>
        </div>
      ) : (
        <p className="form-hint">
          当前视频：<strong>{initialVideo.name}</strong>
          {initialVideo.duration_seconds ? `（${initialVideo.duration_seconds.toFixed(1)} 秒）` : ''}
        </p>
      )}

      {!initialVideo && sourceMode === 'upload' ? (
        <FormField label="选择长参考视频" hint="支持 MP4、MOV、MKV、WebM（最大 1 GiB，支持 3–5 分钟整片）。">
          <input
            key={fileInputKey}
            aria-label="上传长参考视频"
            type="file"
            accept=".mp4,.mov,.mkv,.webm,video/*"
            disabled={Boolean(busy) || disabled}
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
        </FormField>
      ) : null}

      {!initialVideo && sourceMode === 'library' ? (
        <FormField label="选择已有参考视频">
          <select
            aria-label="选择参考视频库视频"
            value={videoId}
            disabled={Boolean(busy) || disabled}
            onChange={e => {
              setVideoId(e.target.value);
              setPreview(null);
            }}
          >
            <option value="">请选择视频…</option>
            {libraryVideos.map(v => (
              <option key={v.id} value={v.id}>
                {v.name}（{v.metadata?.duration_seconds?.toFixed(1) ?? '—'} 秒）
              </option>
            ))}
          </select>
        </FormField>
      ) : null}

      <div className="workspace-grid">
        <FormField label="新项目名称（可选）">
          <input
            value={projectName}
            maxLength={200}
            placeholder="留空则按视频名称自动命名"
            disabled={Boolean(busy) || disabled}
            onChange={e => setProjectName(e.target.value)}
          />
        </FormField>
        <FormField label="参考用途">
          <select
            value={referenceRole}
            disabled={Boolean(busy) || disabled}
            onChange={e => setReferenceRole(e.target.value)}
          >
            {ROLE_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </FormField>
        <FormField label="单镜最小时长（秒）">
          <input
            type="number"
            min={1}
            max={10}
            step={0.5}
            value={minShot}
            disabled={Boolean(busy) || disabled}
            onChange={e => setMinShot(Number(e.target.value))}
          />
        </FormField>
        <FormField label="单镜目标时长（秒）">
          <input
            type="number"
            min={2}
            max={12}
            step={0.5}
            value={targetShot}
            disabled={Boolean(busy) || disabled}
            onChange={e => setTargetShot(Number(e.target.value))}
          />
        </FormField>
        <FormField label="单镜最大时长（秒，建议≤8）">
          <input
            type="number"
            min={2}
            max={15}
            step={0.5}
            value={maxShot}
            disabled={Boolean(busy) || disabled}
            onChange={e => setMaxShot(Number(e.target.value))}
          />
        </FormField>
        <FormField label="转场敏感度（0.05–0.95）">
          <input
            type="number"
            min={0.05}
            max={0.95}
            step={0.05}
            value={sceneThreshold}
            disabled={Boolean(busy) || disabled}
            onChange={e => setSceneThreshold(Number(e.target.value))}
          />
        </FormField>
      </div>

      <FormField label="统一主体/风格约束（可选，附加到每个分镜提示词前）">
        <input
          value={customPromptPrefix}
          maxLength={1200}
          placeholder="例如：写实电影光影，单人自然动作，固定商品外观不变形"
          disabled={Boolean(busy) || disabled}
          onChange={e => setCustomPromptPrefix(e.target.value)}
        />
      </FormField>

      <label className="studio-check">
        <input
          type="checkbox"
          checked={analyzeWithVlm}
          disabled={Boolean(busy) || disabled}
          onChange={e => setAnalyzeWithVlm(e.target.checked)}
        />
        <span>切分时调用本机视觉模型（Qwen3-VL）逐镜反推画面描述（耗时较长；未勾选时秒级切分建项目）</span>
      </label>

      <label className="studio-check">
        <input
          type="checkbox"
          checked={includeAudio}
          disabled={Boolean(busy) || disabled}
          onChange={e => setIncludeAudio(e.target.checked)}
        />
        <span>将参考片段原音轨作为节奏/环境参考（默认关闭，仅参考视觉画面）</span>
      </label>

      {busy ? <p role="status">{busy}</p> : null}
      {error ? <p role="alert" className="inline-notice warning">{error}</p> : null}

      {preview ? (
        <div className="inline-notice" role="status">
          <strong>切分预览：{preview.shot_count} 个分镜</strong>
          <p>
            原片 {preview.source_duration_seconds} 秒（画幅 {preview.aspect_ratio}），检测到 {preview.detected_scene_cuts} 处场景段落，规划总时长 {preview.total_planned_duration_seconds} 秒。
          </p>
        </div>
      ) : null}

      {result ? (
        <div className="inline-notice" role="status">
          <strong>已创建多镜复刻项目：{result.project_name}</strong>
          <p>
            共拆分 {result.shot_count} 镜（原片 {result.source_duration_seconds} 秒，画幅 {result.aspect_ratio}，规划时长 {result.total_planned_duration_seconds} 秒
            {result.vlm_described_count ? `，本机 VLM 反推 ${result.vlm_described_count} 镜` : ''}）。
            可在下方绑定人物/产品参考图后点击「开始／增量续作」。
          </p>
        </div>
      ) : null}

      <div className="studio-actions">
        {(initialVideo || sourceMode === 'library') ? (
          <button
            type="button"
            className="button button-secondary"
            disabled={!canSubmit}
            onClick={() => void handlePreview()}
          >
            <Scissors size={15} />预览切分方案
          </button>
        ) : null}
        <button type="submit" className="button button-primary" disabled={!canSubmit}>
          <Film size={15} />{busy ? '正在切分建项目…' : '自动切分并创建多镜项目'}
        </button>
      </div>
    </form>
  );

  if (initialVideo) {
    return (
      <details className="workspace-details" open={open} onToggle={e => handleToggle((e.target as HTMLDetailsElement).open)}>
        <summary>按镜头切分复刻为多镜项目（适配 3–5 分钟长视频）</summary>
        {body}
      </details>
    );
  }

  return (
    <details
      className="card studio-panel"
      aria-label="长参考视频一键切片复刻"
      open={open}
      onToggle={e => handleToggle((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        <strong>长参考视频一键切片复刻（3–5 分钟整片自动拆分为多镜项目）</strong>
      </summary>
      {body}
    </details>
  );
}
