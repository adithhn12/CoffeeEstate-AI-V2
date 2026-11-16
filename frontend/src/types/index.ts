export interface DetectionBox {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [xc, yc, w, h] normalized
  pixel_coords?: [number, number, number, number]; // [x1, y1, x2, y2]
}

export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface ImageAnalysis {
  id: number;
  user_id?: number | null;
  section_id?: number | null;
  section_name?: string;
  section_code?: string;
  original_filename: string;
  storage_path: string;
  annotated_storage_path?: string;
  uploaded_at: string;
  analyzed_at: string;
  analysis_mode?: 'branch' | 'whole_plant';
  model_version: string;
  model_status_at_inference: string;
  inference_time_ms: number;
  confidence_threshold: number;
  total_cherries: number;
  dark_brown_count: number;
  green_count: number;
  low_visibility_count: number;
  red_count: number;
  yellow_count: number;
  dark_brown_pct: number;
  green_pct: number;
  low_visibility_pct: number;
  red_pct: number;
  yellow_pct: number;
  maturity_stage: string;
  maturity_notes?: string;
  detections?: DetectionBox[];
  is_sample_data: boolean;

  // Enriched Maturity Fields
  evaluable_cherries?: number;
  evaluable_red_pct?: number;
  evaluable_yellow_pct?: number;
  evaluable_green_pct?: number;
  evaluable_dark_brown_pct?: number;
  uncertain_cherries?: number;
  uncertain_pct?: number;
  mean_confidence?: number;
  denominator_note?: string;
  yield_disclaimer?: string;
  plant_ripeness_score?: number;
  tiles_processed?: number;
  quality_assessment?: string;
}

export interface BatchAnalysisResponse {
  section_id?: number | null;
  section_name?: string;
  total_images_analyzed: number;
  aggregated_total_cherries: number;
  avg_red_pct: number;
  avg_yellow_pct: number;
  avg_green_pct: number;
  avg_dark_brown_pct: number;
  avg_low_visibility_pct: number;
  overall_maturity_stage: string;
  maturity_recommendation: string;
  individual_results: ImageAnalysis[];

  // Enriched Batch Fields
  aggregated_evaluable_cherries?: number;
  avg_evaluable_red_pct?: number;
  avg_evaluable_yellow_pct?: number;
  avg_evaluable_green_pct?: number;
  avg_evaluable_dark_brown_pct?: number;
  avg_uncertain_pct?: number;
  photographic_yield_disclaimer?: string;
  plant_ripeness_score?: number;
}

export interface ModelStatusInfo {
  model_name: string;
  model_version: string;
  weights_path: string;
  weights_exist: boolean;
  device: string;
  cuda_available: boolean;
  status_label: string;
  status_badge_color: string;
  class_mappings: Record<number, string>;
  dataset_info: {
    source: string;
    train_images: number;
    val_images: number;
    classes: number;
  };
  scientific_disclaimer: string;
  detection_parameters?: {
    confidence_threshold: number;
    nms_iou_threshold: number;
    max_detections: number;
    input_resolution: number;
  };
}

export interface HealthStatus {
  status: string;
  database_connected: boolean;
  model_status: string;
  device: string;
  cuda_available: boolean;
  app_name: string;
  version: string;
}
