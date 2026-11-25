import React, { useState, useEffect } from 'react';
import {
  RefreshCw,
  Layers,
  Box,
} from 'lucide-react';
import { api } from '../api/client';
import { ModelStatusInfo } from '../types';

export const SettingsModel: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatusInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const mStatus = await api.getModelStatus().catch(() => null);
      setModelStatus(mStatus);
    } catch (err) {
      console.error('Error loading model info:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  // Class definitions with biological meaning
  const classDescriptions: Record<string, { label: string; desc: string; color: string; badgeBg: string }> = {
    '0': {
      label: 'Dark Brown Cherry',
      desc: 'Overripe / dried — harvest immediately to prevent mold and processing defects',
      color: '#78350F',
      badgeBg: 'bg-stone-100 text-stone-800 border-stone-300',
    },
    '1': {
      label: 'Green Cherry',
      desc: 'Unripe — requires additional weeks of maturation; plucking reduces cup quality',
      color: '#10B981',
      badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-300',
    },
    '2': {
      label: 'Low Visibility / Unsure',
      desc: 'Occluded or shaded — excluded from ripeness denominator to avoid skewing',
      color: '#06B6D4',
      badgeBg: 'bg-sky-50 text-sky-800 border-sky-300',
    },
    '3': {
      label: 'Red Cherry',
      desc: 'Prime ripe — target harvest window for premium washed Arabica/Robusta processing',
      color: '#EF4444',
      badgeBg: 'bg-rose-50 text-rose-800 border-rose-300',
    },
    '4': {
      label: 'Yellow Cherry',
      desc: 'Semi-ripe turning phase — approaching harvest window within 5–10 days',
      color: '#F59E0B',
      badgeBg: 'bg-amber-50 text-amber-800 border-amber-300',
    },
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto pb-12">
      {/* ================= PAGE HEADER ================= */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-stone-900 tracking-tight">
            Model Information
          </h1>
          <p className="text-sm text-stone-600 mt-1">
            Verified details about the coffee cherry detection model and its configuration.
          </p>
        </div>

        <button
          onClick={handleRefresh}
          disabled={refreshing || loading}
          className="btn-touch-secondary py-2 px-3 text-xs flex items-center gap-1.5 self-start sm:self-auto shrink-0"
          title="Reload model information"
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          {refreshing ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {loading && (
        <div className="card-mobile p-6 text-center text-stone-500 text-sm">
          Loading model information…
        </div>
      )}

      {!loading && (
        <>
          {/* ================= 1. MODEL OVERVIEW ================= */}
          <div className="card-mobile space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-stone-100">
              <Box size={18} className="text-estate-800" />
              <h2 className="text-base font-bold text-stone-900">Model Overview</h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              {/* Model Name */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Model Name</span>
                <span className="font-semibold text-stone-900 block">
                  {modelStatus?.model_name || 'Coffee Cherry Detector'}
                </span>
              </div>

              {/* Architecture */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Architecture</span>
                <span className="font-semibold text-stone-900 block font-mono">
                  {modelStatus?.active_architecture || 'Faster R-CNN (ResNet50-FPN)'}
                </span>
              </div>

              {/* Task */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Task</span>
                <span className="font-semibold text-stone-900 block">
                  Object detection — 5 coffee cherry categories
                </span>
              </div>

              {/* Training Dataset */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Training Dataset</span>
                <span className="font-semibold text-stone-900 block">
                  {modelStatus?.dataset_info?.source || 'Croppie Coffee Uganda'}
                </span>
              </div>

              {/* Dataset Split */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Dataset Split</span>
                {modelStatus?.dataset_info ? (
                  <span className="font-semibold text-stone-900 block">
                    {modelStatus.dataset_info.train_images.toLocaleString()} train /{' '}
                    {modelStatus.dataset_info.val_images.toLocaleString()} validation images
                  </span>
                ) : (
                  <span className="text-stone-400 italic text-[11px]">Unavailable</span>
                )}
              </div>

              {/* Model version */}
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200">
                <span className="text-stone-500 block text-[11px] mb-1">Model Version Tag</span>
                <span className="font-mono font-semibold text-stone-900 block text-[11px]">
                  {modelStatus?.model_version || '—'}
                </span>
              </div>
            </div>
          </div>

          {/* ================= 2. CLASS LABELS ================= */}
          <div className="card-mobile space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-stone-100">
              <div className="flex items-center gap-2">
                <Layers size={18} className="text-estate-800" />
                <h2 className="text-base font-bold text-stone-900">
                  Class Labels (
                  {modelStatus?.class_mappings ? Object.keys(modelStatus.class_mappings).length : 5} classes)
                </h2>
              </div>
              <span className="text-xs text-stone-400">Color-coded bounding box categories</span>
            </div>

            <div className="space-y-2">
              {Object.entries(
                modelStatus?.class_mappings || {
                  '0': 'dark_brown_cherry',
                  '1': 'green_cherry',
                  '2': 'low_visibility_unsure',
                  '3': 'red_cherry',
                  '4': 'yellow_cherry',
                }
              ).map(([classId, rawName]) => {
                const info = classDescriptions[classId] || {
                  label: rawName,
                  desc: 'Detection class',
                  color: '#15803d',
                  badgeBg: 'bg-stone-100 text-stone-800 border-stone-200',
                };

                return (
                  <div
                    key={classId}
                    className="p-3 rounded-xl border border-stone-200 bg-white flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
                  >
                    <div className="flex items-center gap-2.5">
                      <span
                        className="w-3.5 h-3.5 rounded-full shrink-0 shadow-2xs border border-white"
                        style={{ backgroundColor: info.color }}
                      />
                      <div>
                        <span className="font-bold text-stone-900">{info.label}</span>
                        <span className="text-stone-400 font-mono text-[11px] ml-1.5">
                          (ID: {classId})
                        </span>
                        <p className="text-[11px] text-stone-500 mt-0.5">{info.desc}</p>
                      </div>
                    </div>

                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-medium border self-start sm:self-auto shrink-0 font-mono ${info.badgeBg}`}
                    >
                      {rawName}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
