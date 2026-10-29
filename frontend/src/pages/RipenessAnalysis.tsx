import React, { useState, useEffect, useRef } from 'react';
import {
  ScanEye,
  Camera,
  Image as ImageIcon,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sliders,
  Trash2,
  ChevronDown,
  ChevronUp,
  X,
  Info,
  ZoomIn,
  TrendingUp,
  Droplets,
  Calendar,
  AlertTriangle,
  ClipboardList
} from 'lucide-react';
import { api } from '../api/client';
import { ImageAnalysis, BatchAnalysisResponse } from '../types';

const MAX_HISTORY_DISPLAY = 5;

export const RipenessAnalysis: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [filePreviews, setFilePreviews] = useState<string[]>([]);
  const [confidence, setConfidence] = useState<number>(0.25);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<BatchAnalysisResponse | null>(null);
  const [history, setHistory] = useState<ImageAnalysis[]>([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<ImageAnalysis | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [zoomModalOpen, setZoomModalOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const h = await api.getAnalysisHistory(undefined);
      const sorted = [...h].sort(
        (a, b) => new Date(b.analyzed_at).getTime() - new Date(a.analyzed_at).getTime()
      );
      const recent = sorted.slice(0, MAX_HISTORY_DISPLAY);
      setHistory(recent);
      if (recent.length > 0 && !selectedHistoryItem && !analysisResult) {
        setSelectedHistoryItem(recent[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);
    setSelectedFiles(files);

    const urls = files.map((f) => URL.createObjectURL(f));
    setFilePreviews(urls);
    setErrorMsg(null);
  };

  const handleRemoveSelectedFile = (idx: number) => {
    const updatedFiles = selectedFiles.filter((_, i) => i !== idx);
    const updatedUrls = filePreviews.filter((_, i) => i !== idx);
    setSelectedFiles(updatedFiles);
    setFilePreviews(updatedUrls);
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) {
      setErrorMsg('Please select or take at least one coffee cherry photo.');
      return;
    }

    try {
      setIsAnalyzing(true);
      setErrorMsg(null);
      const res = await api.uploadAndAnalyze(null, selectedFiles, confidence);
      setAnalysisResult(res);
      if (res.individual_results.length > 0) {
        setSelectedHistoryItem(res.individual_results[0]);
      }
      loadHistory();
    } catch (err: any) {
      setErrorMsg(err.message || 'Analysis failed. Please check image format.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDeleteHistory = async (id: number) => {
    if (!confirm('Are you sure you want to remove this analysis record?')) return;
    try {
      await api.deleteAnalysis(id);
      if (selectedHistoryItem?.id === id) {
        setSelectedHistoryItem(null);
      }
      loadHistory();
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  const activeDisplayItem = selectedHistoryItem;

  // Derive comprehensive recommendations from cherry distribution
  const getExtendedRecommendations = (item: ImageAnalysis) => {
    const red = item.red_pct || 0;
    const yellow = item.yellow_pct || 0;
    const green = item.green_pct || 0;
    const brown = item.dark_brown_pct || 0;
    const uncertain = item.low_visibility_pct || 0;

    const ripenessScore = Math.min(100, Math.round((red * 1.0) + (yellow * 0.45)));

    let harvestMethod = 'Selective Hand-Picking';
    let harvestWindow = 'Monitor development every 4–6 days';
    let processingRoute = 'Standard processing with flotation separation';
    let sugarPotential = 'Developing (Moderate)';
    let urgencyBadge = 'bg-stone-100 text-stone-700 border-stone-300';
    let urgencyLabel = 'Standard Monitoring';

    if (brown > 20) {
      harvestMethod = 'Sanitation Strip Picking & Separation';
      harvestWindow = 'Immediate action required';
      processingRoute = 'Natural (Dry) processing — floaters segregated';
      sugarPotential = 'Degrading (Overripe / Dehydrated)';
      urgencyBadge = 'bg-amber-100 text-amber-900 border-amber-300';
      urgencyLabel = 'High Urgency — Overripe Risk';
    } else if (red >= 55) {
      harvestMethod = 'Selective Picking (Prime Crimson Cherries)';
      harvestWindow = 'Immediate window (Next 2–5 days for peak quality)';
      processingRoute = 'Washed (Wet) processing for premium parchment';
      sugarPotential = 'Optimal (Peak Brix & Mucilage Content)';
      urgencyBadge = 'bg-emerald-100 text-emerald-900 border-emerald-300';
      urgencyLabel = 'Peak Harvest Window';
    } else if ((red + yellow >= 55) && red >= 30) {
      harvestMethod = 'Fly Picking (First Selective Pass)';
      harvestWindow = 'Begin initial pass now; secondary pass in 7–10 days';
      processingRoute = 'Washed or Honey process (segregate turning fruit)';
      sugarPotential = 'High in red fruit, developing in yellow';
      urgencyBadge = 'bg-amber-50 text-amber-900 border-amber-300';
      urgencyLabel = 'Fly Picking Ready';
    } else if (green >= 50) {
      harvestMethod = 'Delay Picking — Allow Maturation';
      harvestWindow = 'Hold harvest for 2–4 weeks';
      processingRoute = 'Not ready for processing';
      sugarPotential = 'Low (Immature sugars, high acidity)';
      urgencyBadge = 'bg-stone-100 text-stone-700 border-stone-300';
      urgencyLabel = 'Immature / Hold';
    }

    const actionableSteps: string[] = [];
    if (red >= 50) {
      actionableSteps.push('Depulp picked cherries within 6 to 8 hours to avoid spontaneous sour fermentation.');
      actionableSteps.push('Calibrate pulper drum clearance to prevent nicking ripe cherries.');
    }
    if (green > 10) {
      actionableSteps.push('Instruct pickers to avoid stripping entire clusters; leave green cherries for subsequent rounds.');
    }
    if (yellow >= 15) {
      actionableSteps.push('Schedule a follow-up selective picking round in 7 to 10 days for turning fruit.');
    }
    if (brown > 5) {
      actionableSteps.push('Utilize water flotation tank to skim off floaters and dried cherries before pulping.');
    }
    if (uncertain > 20) {
      actionableSteps.push('High canopy shadow detected; sample branches from opposite slope side for confirmation.');
    }
    if (actionableSteps.length === 0) {
      actionableSteps.push('Continue routine canopy surveillance as fruit maturation progresses.');
    }

    return {
      ripenessScore,
      harvestMethod,
      harvestWindow,
      processingRoute,
      sugarPotential,
      urgencyBadge,
      urgencyLabel,
      actionableSteps
    };
  };

  const recommendations = activeDisplayItem ? getExtendedRecommendations(activeDisplayItem) : null;

  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* ================= 1. PAGE TITLE & EXPLANATION ================= */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-stone-900 tracking-tight">
          Coffee Cherry Ripeness Analysis
        </h1>
        <p className="text-sm text-stone-600 mt-1">
          Upload or capture branch photos to evaluate cherry maturation and receive harvesting recommendations.
        </p>
      </div>

      {/* ================= 2. UPLOAD & INPUT FORM ================= */}
      <div className="card-mobile p-4 sm:p-6 space-y-4">
        {/* Image Upload Area */}
        <div>
          <label className="block text-xs font-bold text-stone-700 uppercase tracking-wider mb-1.5">
            Coffee Cherry Photos *
          </label>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            capture="environment"
            multiple
            onChange={handleFileChange}
            className="hidden"
          />

          {filePreviews.length === 0 ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-stone-300 hover:border-estate-700 bg-estate-50/40 rounded-2xl p-6 text-center cursor-pointer transition-colors active:bg-estate-100/50"
            >
              <div className="w-14 h-14 rounded-2xl bg-white text-estate-900 shadow-xs flex items-center justify-center mx-auto mb-3 border border-stone-200">
                <Camera size={28} />
              </div>
              <div className="font-bold text-sm text-stone-900">
                Tap to Take Photo or Choose Files
              </div>
              <p className="text-xs text-stone-500 mt-1">
                Supports clear branch photos (JPG, PNG up to 15MB)
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Previews Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                {filePreviews.map((url, idx) => (
                  <div key={idx} className="relative rounded-xl overflow-hidden border border-stone-200 aspect-square bg-stone-100">
                    <img src={url} alt={`Preview ${idx}`} className="w-full h-full object-cover" />
                    <button
                      type="button"
                      onClick={() => handleRemoveSelectedFile(idx)}
                      className="absolute top-1.5 right-1.5 w-7 h-7 rounded-full bg-black/70 text-white flex items-center justify-center hover:bg-black"
                      title="Remove image"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-touch-secondary flex-1 text-xs py-2.5 min-h-[42px]"
                >
                  <Camera size={15} /> Add More Photos
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedFiles([]);
                    setFilePreviews([]);
                  }}
                  className="btn-touch-secondary text-xs py-2.5 px-3 min-h-[42px] text-stone-500 hover:text-rose-700"
                >
                  Clear All
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Expandable Advanced Settings (Confidence Threshold) */}
        <div className="border-t border-stone-100 pt-3">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center justify-between w-full text-xs font-semibold text-stone-600 hover:text-stone-900 py-1"
          >
            <span className="flex items-center gap-1.5">
              <Sliders size={14} /> Advanced Settings
            </span>
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {showAdvanced && (
            <div className="mt-3 p-3.5 bg-stone-50 rounded-xl space-y-2 text-xs">
              <div className="flex justify-between items-center font-medium text-stone-700">
                <span>Detection Sensitivity</span>
                <span className="font-mono font-bold text-estate-900">{Math.round(confidence * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-estate-800 cursor-pointer h-2 bg-stone-200 rounded-lg"
              />
              <p className="text-[11px] text-stone-500">
                Default: 25%. Lower values help detect cherries in heavy shade; higher values reduce uncertainty.
              </p>
            </div>
          )}
        </div>

        {/* Error message */}
        {errorMsg && (
          <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-center gap-2">
            <AlertCircle size={16} className="shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={isAnalyzing || selectedFiles.length === 0}
          className="btn-touch-primary w-full text-base font-bold shadow-sm"
        >
          {isAnalyzing ? (
            <span className="flex items-center gap-2">
              <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Analyzing Cherries...
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <ScanEye size={20} />
              Analyse Cherry Ripeness
            </span>
          )}
        </button>
      </div>

      {/* ================= 3. PREDICTION RESULTS ================= */}
      {activeDisplayItem ? (
        <div className="card-mobile p-4 sm:p-6 space-y-5 border-2 border-estate-200">
          {/* Result Header */}
          <div className="flex items-center justify-between pb-3 border-b border-stone-200">
            <div>
              <div className="flex items-center gap-2">
                <span className="badge-sage text-[11px]">Ripeness Analysis Result</span>
                {recommendations && (
                  <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${recommendations.urgencyBadge}`}>
                    {recommendations.urgencyLabel}
                  </span>
                )}
                <span className="text-xs font-mono text-stone-400">
                  {activeDisplayItem.inference_time_ms ? `${activeDisplayItem.inference_time_ms} ms` : ''}
                </span>
              </div>
              <h2 className="text-lg font-bold text-stone-900 mt-0.5">
                {activeDisplayItem.maturity_stage || 'Optimal Harvest Window'}
              </h2>
            </div>

            <button
              onClick={() => setZoomModalOpen(true)}
              className="btn-touch-secondary py-2 px-3 text-xs flex items-center gap-1.5"
              title="Open full image"
            >
              <ZoomIn size={15} /> Zoom Photo
            </button>
          </div>

          {/* Full Aspect-Ratio Annotated Prediction Image */}
          <div
            onClick={() => setZoomModalOpen(true)}
            className="relative rounded-2xl overflow-hidden bg-stone-950 border border-stone-200 cursor-pointer group shadow-inner"
          >
            {activeDisplayItem.annotated_storage_path ? (
              <img
                src={`/${activeDisplayItem.annotated_storage_path}`}
                alt="Detected Cherries"
                className="w-full h-auto max-h-[550px] object-contain mx-auto"
              />
            ) : (
              <img
                src={`/${activeDisplayItem.storage_path}`}
                alt="Original Cherries"
                className="w-full h-auto max-h-[550px] object-contain mx-auto"
              />
            )}
            <div className="absolute bottom-2.5 right-2.5 bg-black/75 backdrop-blur-md text-white text-[11px] px-2.5 py-1 rounded-lg font-mono flex items-center gap-1.5">
              <ZoomIn size={12} /> Tap to zoom
            </div>
          </div>

          {/* Ripeness Distribution Progress Bar */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-stone-700 uppercase tracking-wider text-[11px]">Ripeness Distribution</span>
              {recommendations && (
                <span className="text-stone-600 font-semibold text-[11px]">
                  Maturity Index: <strong className="text-estate-900">{recommendations.ripenessScore}/100</strong>
                </span>
              )}
            </div>
            <div className="w-full h-3.5 bg-stone-100 rounded-full overflow-hidden flex shadow-inner">
              {activeDisplayItem.red_pct > 0 && (
                <div style={{ width: `${activeDisplayItem.red_pct}%` }} className="bg-rose-500 h-full" title={`Red (Ripe): ${activeDisplayItem.red_pct}%`} />
              )}
              {activeDisplayItem.yellow_pct > 0 && (
                <div style={{ width: `${activeDisplayItem.yellow_pct}%` }} className="bg-amber-400 h-full" title={`Yellow (Semi-ripe): ${activeDisplayItem.yellow_pct}%`} />
              )}
              {activeDisplayItem.green_pct > 0 && (
                <div style={{ width: `${activeDisplayItem.green_pct}%` }} className="bg-emerald-500 h-full" title={`Green (Unripe): ${activeDisplayItem.green_pct}%`} />
              )}
              {activeDisplayItem.dark_brown_pct > 0 && (
                <div style={{ width: `${activeDisplayItem.dark_brown_pct}%` }} className="bg-amber-900 h-full" title={`Dark Brown (Overripe): ${activeDisplayItem.dark_brown_pct}%`} />
              )}
              {activeDisplayItem.low_visibility_pct > 0 && (
                <div style={{ width: `${activeDisplayItem.low_visibility_pct}%` }} className="bg-sky-400 h-full" title={`Uncertain / Occluded: ${activeDisplayItem.low_visibility_pct}%`} />
              )}
            </div>
          </div>

          {/* Class-wise Count Cards with Friendly Labels */}
          <div>
            <div className="text-xs font-bold text-stone-700 uppercase tracking-wider mb-2">
              Detected Cherries ({activeDisplayItem.total_cherries} total)
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center">
              {/* Red (Ripe) */}
              <div className="p-3 bg-rose-50/80 border border-rose-200 rounded-xl">
                <span className="text-[10px] font-bold text-rose-800 uppercase block">Red (Ripe)</span>
                <div className="text-xl font-bold text-rose-700 mt-0.5">{activeDisplayItem.red_count}</div>
                <span className="text-xs text-rose-600 font-semibold">{activeDisplayItem.red_pct}%</span>
              </div>

              {/* Yellow (Turning) */}
              <div className="p-3 bg-amber-50/80 border border-amber-200 rounded-xl">
                <span className="text-[10px] font-bold text-amber-800 uppercase block">Yellow (Semi)</span>
                <div className="text-xl font-bold text-amber-700 mt-0.5">{activeDisplayItem.yellow_count}</div>
                <span className="text-xs text-amber-600 font-semibold">{activeDisplayItem.yellow_pct}%</span>
              </div>

              {/* Green (Unripe) */}
              <div className="p-3 bg-emerald-50/80 border border-emerald-200 rounded-xl">
                <span className="text-[10px] font-bold text-emerald-800 uppercase block">Green (Unripe)</span>
                <div className="text-xl font-bold text-emerald-700 mt-0.5">{activeDisplayItem.green_count}</div>
                <span className="text-xs text-emerald-600 font-semibold">{activeDisplayItem.green_pct}%</span>
              </div>

              {/* Dark Brown */}
              <div className="p-3 bg-stone-100 border border-stone-200 rounded-xl">
                <span className="text-[10px] font-bold text-stone-700 uppercase block">Dark Brown</span>
                <div className="text-xl font-bold text-stone-800 mt-0.5">{activeDisplayItem.dark_brown_count}</div>
                <span className="text-xs text-stone-600 font-semibold">{activeDisplayItem.dark_brown_pct}%</span>
              </div>

              {/* Uncertain / Low Visibility */}
              <div className="p-3 bg-sky-50 border border-sky-200 rounded-xl col-span-2 sm:col-span-1">
                <span className="text-[10px] font-bold text-sky-800 uppercase block">Uncertain</span>
                <div className="text-xl font-bold text-sky-700 mt-0.5">{activeDisplayItem.low_visibility_count}</div>
                <span className="text-xs text-sky-600 font-semibold">{activeDisplayItem.low_visibility_pct}%</span>
              </div>
            </div>
          </div>

          {/* Evaluable Ripeness Summary */}
          {activeDisplayItem.evaluable_cherries !== undefined && activeDisplayItem.evaluable_cherries !== null && (
            <div className="p-3.5 bg-estate-50/80 border border-estate-200 rounded-xl text-xs space-y-1.5">
              <div className="flex items-center justify-between text-[11px] font-semibold text-estate-950">
                <span>Evaluable Base ({activeDisplayItem.evaluable_cherries} discernible cherries)</span>
                <span className="text-sky-700 font-medium">Uncertain / Shaded: {activeDisplayItem.low_visibility_count}</span>
              </div>
              <div className="flex items-center gap-3 text-stone-700 flex-wrap text-[11px]">
                <span className="text-rose-700 font-semibold">Ripe Red: {activeDisplayItem.evaluable_red_pct ?? activeDisplayItem.red_pct}%</span>
                <span className="text-amber-700 font-semibold">Semi Yellow: {activeDisplayItem.evaluable_yellow_pct ?? activeDisplayItem.yellow_pct}%</span>
                <span className="text-emerald-700 font-semibold">Unripe Green: {activeDisplayItem.evaluable_green_pct ?? activeDisplayItem.green_pct}%</span>
                <span className="text-stone-700 font-semibold">Overripe Brown: {activeDisplayItem.evaluable_dark_brown_pct ?? activeDisplayItem.dark_brown_pct}%</span>
              </div>
            </div>
          )}

          {/* Actionable Harvest Recommendations Grid */}
          {recommendations && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200 space-y-1">
                <div className="text-stone-500 flex items-center gap-1.5 font-medium text-[11px]">
                  <Calendar size={13} className="text-estate-700" />
                  Target Harvest Window
                </div>
                <div className="font-bold text-stone-900 leading-snug">
                  {recommendations.harvestWindow}
                </div>
              </div>

              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200 space-y-1">
                <div className="text-stone-500 flex items-center gap-1.5 font-medium text-[11px]">
                  <ClipboardList size={13} className="text-estate-700" />
                  Recommended Method
                </div>
                <div className="font-bold text-stone-900 leading-snug">
                  {recommendations.harvestMethod}
                </div>
              </div>

              <div className="p-3 bg-stone-50 rounded-xl border border-stone-200 space-y-1">
                <div className="text-stone-500 flex items-center gap-1.5 font-medium text-[11px]">
                  <Droplets size={13} className="text-estate-700" />
                  Processing Route
                </div>
                <div className="font-bold text-stone-900 leading-snug">
                  {recommendations.processingRoute}
                </div>
              </div>
            </div>
          )}

          {/* Practical Agronomic Guidance & Field Next Steps */}
          <div className="p-4 bg-stone-50 border border-stone-200 rounded-2xl text-xs space-y-3">
            <div className="font-bold text-stone-900 flex items-center gap-1.5">
              <CheckCircle2 size={16} className="text-estate-800" />
              Maturity Interpretation & Picking Advice:
            </div>
            <p className="text-stone-700 leading-relaxed whitespace-pre-line text-xs sm:text-sm">
              {activeDisplayItem.maturity_notes ||
                'Coffee cherries evaluated according to standard harvest guidelines.'}
            </p>

            {/* Field Action Checklist */}
            {recommendations && recommendations.actionableSteps.length > 0 && (
              <div className="border-t border-stone-200 pt-3 space-y-2">
                <span className="font-bold text-stone-800 text-[11px] uppercase tracking-wider block">
                  Recommended Action Steps:
                </span>
                <ul className="space-y-1.5 text-stone-700">
                  {recommendations.actionableSteps.map((step, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-xs">
                      <span className="text-emerald-700 font-bold shrink-0 mt-0.5">•</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="text-[11px] text-stone-500 border-t border-stone-200/80 pt-2 flex items-start gap-1.5">
              <Info size={13} className="shrink-0 text-stone-400 mt-0.5" />
              <span>
                <strong>Field Note:</strong> AI predictions evaluate sampled canopy branches to indicate ripeness timing. Verify physical clusters across terraces before mobilizing harvest labor.
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="card-mobile p-8 text-center text-stone-400 flex flex-col items-center justify-center min-h-[220px]">
          <ScanEye size={40} className="text-stone-300 mb-2" />
          <h3 className="text-base font-semibold text-stone-700">No Image Selected</h3>
          <p className="text-xs text-stone-500 max-w-sm mt-1">
            Take or upload a coffee photo above to view bounding box detections, maturity distribution, and harvesting recommendations.
          </p>
        </div>
      )}

      {/* ================= 4. ANALYSIS HISTORY (max 5) ================= */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold text-stone-900 uppercase tracking-wider">
            Recent Analyses {history.length > 0 ? `(${history.length})` : ''}
          </h2>
          <span className="text-xs text-stone-500">Tap to review past photo</span>
        </div>

        {history.length > 0 ? (
          <div className="space-y-2.5">
            {history.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedHistoryItem(item)}
                className={`card-mobile p-3 flex items-center justify-between gap-3 cursor-pointer transition-all ${
                  selectedHistoryItem?.id === item.id ? 'border-estate-800 bg-estate-50/30' : 'hover:border-stone-300'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-14 h-14 rounded-xl bg-stone-100 overflow-hidden shrink-0 border border-stone-200">
                    <img
                      src={`/${item.annotated_storage_path || item.storage_path}`}
                      alt="Thumbnail"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold text-stone-900 truncate">
                      {item.original_filename}
                    </div>
                    <div className="text-[11px] text-stone-500 flex items-center gap-2 mt-0.5">
                      <span>{item.total_cherries} cherries</span>
                      <span>•</span>
                      <span className="text-rose-700 font-semibold">{item.red_pct}% Red</span>
                    </div>
                    <div className="text-[10px] text-stone-400 flex items-center gap-1 mt-0.5">
                      <Clock size={10} />
                      {new Date(item.analyzed_at).toLocaleString('en-IN', {
                        day: 'numeric',
                        month: 'short',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="badge-sage text-[10px]">
                    {item.maturity_stage}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteHistory(item.id);
                    }}
                    className="p-1.5 text-stone-400 hover:text-rose-600 rounded-lg"
                    title="Delete record"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="card-mobile p-8 text-center flex flex-col items-center justify-center">
            <ScanEye size={32} className="text-stone-200 mb-2" />
            <p className="text-sm text-stone-500 font-medium">No previous analyses yet</p>
            <p className="text-xs text-stone-400 mt-1">Upload a photo above to run your first analysis.</p>
          </div>
        )}
      </div>

      {/* ================= FULL IMAGE ZOOM MODAL ================= */}
      {zoomModalOpen && activeDisplayItem && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col justify-between p-4">
          <div className="flex items-center justify-between text-white pb-2">
            <div>
              <div className="font-bold text-sm">{activeDisplayItem.original_filename}</div>
              <div className="text-xs text-stone-400">{activeDisplayItem.total_cherries} cherries detected</div>
            </div>
            <button
              onClick={() => setZoomModalOpen(false)}
              className="w-9 h-9 rounded-full bg-white/20 text-white flex items-center justify-center hover:bg-white/30"
            >
              <X size={18} />
            </button>
          </div>

          <div className="flex-1 flex items-center justify-center overflow-auto p-2">
            <img
              src={`/${activeDisplayItem.annotated_storage_path || activeDisplayItem.storage_path}`}
              alt="Zoomed Prediction"
              className="max-w-full max-h-[85vh] object-contain rounded-xl"
            />
          </div>

          <div className="text-center text-xs text-stone-400 pt-2">
            Pinch or zoom on touchscreen to inspect individual cherry clusters
          </div>
        </div>
      )}
    </div>
  );
};
