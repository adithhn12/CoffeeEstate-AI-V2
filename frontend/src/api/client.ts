import {
  ImageAnalysis,
  BatchAnalysisResponse,
  ModelStatusInfo,
  HealthStatus
} from '../types';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || errorJson.message || res.statusText;
    } catch {
      errorDetail = res.statusText;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  // Health
  getHealth: () => fetchJson<HealthStatus>(`${API_BASE}/health`),

  // Model Diagnostics & Specifications
  getModelStatus: () => fetchJson<ModelStatusInfo>(`${API_BASE}/analysis/model-status`),

  // Ripeness Analysis & CV
  uploadAndAnalyze: (
    sectionId: number | null,
    files: File[],
    confidence = 0.25,
    mode: 'branch' | 'whole_plant' = 'branch'
  ) => {
    const formData = new FormData();
    if (sectionId !== null && sectionId !== undefined) {
      formData.append('section_id', sectionId.toString());
    }
    formData.append('confidence_threshold', confidence.toString());
    formData.append('analysis_mode', mode);
    files.forEach((f) => formData.append('files', f));

    return fetchJson<BatchAnalysisResponse>(`${API_BASE}/analysis/upload`, {
      method: 'POST',
      body: formData,
    });
  },

  getAnalysisHistory: (sectionId?: number) =>
    fetchJson<ImageAnalysis[]>(
      sectionId
        ? `${API_BASE}/analysis/history?section_id=${sectionId}`
        : `${API_BASE}/analysis/history`
    ),

  deleteAnalysis: (id: number) =>
    fetchJson<{ message: string }>(`${API_BASE}/analysis/history/${id}`, {
      method: 'DELETE',
    }),

  // Sample data management
  seedSampleData: () =>
    fetchJson<{ message: string }>(`${API_BASE}/sample-data/seed`, { method: 'POST' }),

  clearSampleData: () =>
    fetchJson<{ message: string }>(`${API_BASE}/sample-data/clear`, { method: 'POST' }),
};
