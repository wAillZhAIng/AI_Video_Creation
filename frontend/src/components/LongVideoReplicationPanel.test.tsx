import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { request } from '../lib/api';
import { LongVideoReplicationPanel } from './LongVideoReplicationPanel';

vi.mock('../lib/api', async importOriginal => {
  const original = await importOriginal<typeof import('../lib/api')>();
  return { ...original, request: vi.fn() };
});
const mockRequest = vi.mocked(request);

describe('LongVideoReplicationPanel', () => {
  afterEach(() => {
    cleanup();
    mockRequest.mockReset();
  });

  it('does not fetch library when closed and submits uploaded long video to import-long-video', async () => {
    const user = userEvent.setup();
    const onCreated = vi.fn();
    mockRequest.mockResolvedValueOnce({
      status: 'imported',
      project_id: 'prj_long_1',
      project_name: '4分钟复刻',
      video_id: 'refvideo_1',
      video_name: 'demo.mp4',
      aspect_ratio: '16:9',
      source_duration_seconds: 240,
      detected_scene_cuts: 12,
      shot_count: 42,
      total_planned_duration_seconds: 236,
      reference_role: 'combined',
      include_audio: false,
      vlm_described_count: 0,
      shots: [],
    });

    render(<LongVideoReplicationPanel onCreated={onCreated} />);
    expect(mockRequest).not.toHaveBeenCalled();

    await user.click(screen.getByText(/长参考视频一键切片复刻/));
    const input = screen.getByLabelText('上传长参考视频') as HTMLInputElement;
    const videoFile = new File(['mock-video'], 'demo.mp4', { type: 'video/mp4' });
    await user.upload(input, videoFile);
    await user.click(screen.getByRole('button', { name: '自动切分并创建多镜项目' }));

    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(mockRequest).toHaveBeenCalledWith(
      '/api/studio/import-long-video',
      expect.objectContaining({ method: 'POST', body: expect.any(FormData) }),
    );
    expect(screen.getByText(/共拆分 42 镜/)).toBeInTheDocument();
  });
});
