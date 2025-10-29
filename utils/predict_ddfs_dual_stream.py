# This is the function for generating DDF from dual-stream network predictions
import os
from utils.funs import load_config
from utils.network_dual_stream import create_dual_stream_network
from utils.transform import Transforms, TransformAccumulation
from utils.plot_functions import reference_image_points, read_calib_matrices
from utils.Transf2DDFs import cal_global_ddfs, cal_local_ddfs
import torch
import json

class DualStreamPrediction():
    """
    Dual-Stream Prediction class for generating DDFs from dual-stream model
    """
    
    def __init__(self, parameters, model_name, data_path_calib, model_path, device, w=640, h=480):
        self.parameters = parameters
        self.device = device
        
        # Load previously generated data pairs
        with open(os.path.join(model_path, "data_pairs.json"), "r", encoding='utf-8') as f:
            data_pairs = json.load(f)
        self.data_pairs = torch.tensor(data_pairs).to(self.device)
        self.model_name = model_name
        self.model_path = model_path
        
        # Load calibration matrix
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = read_calib_matrices(data_path_calib)
        self.tform_calib_scale = self.tform_calib_scale.to(self.device)
        self.tform_calib_R_T = self.tform_calib_R_T.to(self.device)
        self.tform_calib = self.tform_calib.to(self.device)
        
        # Point coordinates in image coordinate system (in pixel), all pixel points
        self.image_points = reference_image_points([h, w], [h, w]).to(self.device)
        
        # Transform prediction into 4*4 transformation matrix
        pred_type = getattr(self.parameters, 'PRED_TYPE', 'parameter')
        self.transforms = Transforms(
            pred_type=pred_type,
            num_pairs=self.data_pairs.shape[0],
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_mm=self.tform_calib_scale
        )
        
        # Accumulate transformations
        self.transform_accumulation = TransformAccumulation(
            image_points=self.image_points,
            tform_image_to_tool=self.tform_calib,
            tform_image_mm_to_tool=self.tform_calib_R_T,
            tform_image_pixel_to_image_mm=self.tform_calib_scale
        )
        
        self.pred_dim = self.type_dim(pred_type, self.image_points.shape[1], self.data_pairs.shape[0])
        
        # Use Dual-Stream model architecture
        self.num_samples = getattr(self.parameters, 'NUM_SAMPLES', 7)
        self.model = create_dual_stream_network(
            input_channels=1,  # Grayscale ultrasound
            embed_dim=256,
            num_frames=self.num_samples,
            output_dim=self.pred_dim,
            margin_alpha=0.2,  # α = 0.2 (from Algorithm 1)
            delta=2,           # δ: positive pair threshold
            Delta=4,           # Δ: negative pair threshold
            tau_sim=0.5        # ε: DBSCAN similarity threshold
        ).to(self.device)
        
        # Load the dual-stream model
        checkpoint = torch.load(os.path.join(self.model_path, self.model_name), map_location=torch.device(self.device))
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def type_dim(self, label_pred_type, num_points=None, num_pairs=1):
        type_dim_dict = {
            "transform": 12,
            "parameter": 6,
            "point": num_points * 3,
            "quaternion": 7
        }
        return type_dim_dict[label_pred_type] * num_pairs
    
    def generate_prediction_DDF(self, frames, landmark):
        frames = torch.tensor(frames)[None, ...].to(self.device)
        frames = frames / 255
        landmark = torch.from_numpy(landmark)
        
        # Predict global and local transformations
        transformation_global, transformation_local = self.cal_pred_transformations(frames)
        
        # Global displacement vectors for pixel reconstruction and landmark reconstruction
        pred_global_allpts_DDF, pred_global_landmark_DDF = cal_global_ddfs(
            transformation_global, self.tform_calib_scale.cpu(), self.image_points.cpu(), landmark
        )
        # Local displacement vectors for pixel reconstruction and landmark reconstruction
        pred_local_allpts_DDF, pred_local_landmark_DDF = cal_local_ddfs(
            transformation_local, self.tform_calib_scale.cpu(), self.image_points.cpu(), landmark
        )
        
        return pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF
    
    def cal_pred_transformations(self, frames):
        """
        Predict global and local transformations using dual-stream model
        """
        # Local transformation, i.e., transformation from current frame to the immediate previous frame
        transformation_local = torch.zeros(frames.shape[1] - 1, 4, 4)
        # Global transformation, i.e., transformation from current frame to the first frame
        transformation_global = torch.zeros(frames.shape[1] - 1, 4, 4)
        
        prev_transf = torch.eye(4).to(self.device)
        idx_f0 = 0  # This is the reference frame for network prediction
        Pair_index = 0  # Select which pair of prediction to use
        interval_pred = torch.squeeze(self.data_pairs[Pair_index])[1] - torch.squeeze(self.data_pairs[Pair_index])[0]
        
        while True:
            with torch.no_grad():
                frames_sub = frames[:, idx_f0:idx_f0 + self.num_samples, ...]
                # Dual-stream model returns sequential output (B, T, 6) during inference
                outputs = self.model(frames_sub, training=False, return_contrastive_loss=False)

                # Handle sequential output: take the prediction for the target frame
                if outputs.dim() == 3:  # (B, T, 6)
                    # Take the prediction for the last frame in the sequence
                    output_for_transform = outputs[0, -1, :]  # (6,)
                else:  # (B, 6) - fallback
                    output_for_transform = outputs[0, :]

                # Transform prediction into 4*4 transformation matrix, to be accumulated
                preds_transf = self.transforms(output_for_transform.unsqueeze(0).unsqueeze(0))[0, Pair_index, ...]
                transformation_local[idx_f0] = preds_transf
                # Calculate global transformation
                prev_transf = self.transform_accumulation(prev_transf, preds_transf)
                transformation_global[idx_f0] = prev_transf

            idx_f0 += interval_pred
            if (idx_f0 + self.num_samples) > frames.shape[1]:
                break

        if self.num_samples > 2:
            transformation_local[idx_f0:, ...] = torch.eye(4).expand(transformation_local[idx_f0:, ...].shape[0], -1, -1)
            transformation_global[idx_f0:, ...] = transformation_global[idx_f0 - 1].expand(transformation_global[idx_f0:, ...].shape[0], -1, -1)
        
        return transformation_global, transformation_local


def predict_ddfs_dual_stream(frames, landmark, data_path_calib, device, model_test_number=None):
    """
    Generate DDFs using dual-stream model
    
    Args:
        frames (numpy.ndarray): shape=(N, 480, 640), frames in the scan
        landmark (numpy.ndarray): shape=(100,3), denoting the locations of landmarks
        data_path_calib (str): path to calibration matrix
        device (str): device to use for prediction, either 'cuda' or 'cpu'
        model_test_number (int): specific test number to use (if None, uses latest)

    Returns:
        pred_global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels
        pred_global_landmark_DDF (numpy.ndarray): shape=(3, 100), global DDF for landmark  
        pred_local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels
        pred_local_landmark_DDF (numpy.ndarray): shape=(3, 100), local DDF for landmark
    """
    
    # Path to the trained Dual-Stream model (find the latest test directory)
    base_path = os.getcwd() + '/results'
    dual_stream_dirs = [d for d in os.listdir(base_path) if d.startswith('train_dual_stream_test_')]
    if not dual_stream_dirs:
        raise FileNotFoundError("No dual-stream model directories found in results/")
    
    # Select model directory
    if model_test_number is not None:
        target_dir = f'train_dual_stream_test_{model_test_number}'
        if target_dir not in dual_stream_dirs:
            raise FileNotFoundError(f"Model directory {target_dir} not found")
        model_path = os.path.join(base_path, target_dir)
    else:
        # Get the latest test directory
        latest_test = max(dual_stream_dirs, key=lambda x: int(x.split('_')[-1]))
        model_path = os.path.join(base_path, latest_test)
    
    # Try different model names
    model_names = ['best_model.pth', 'saved_model/best_validation_dist_model', 'saved_model/best_validation_loss_model']
    model_name = None
    for name in model_names:
        if os.path.exists(os.path.join(model_path, name)):
            model_name = name
            break
    
    if model_name is None:
        raise FileNotFoundError(f"No model found in {model_path}")
    
    # Parameters used for training the Dual-Stream model
    config_files = ['test_config.json', 'test_config.txt', 'config.txt']
    parameters = None
    config_data = None

    for config_file in config_files:
        config_path = os.path.join(model_path, config_file)
        if os.path.exists(config_path):
            if config_file.endswith('.json'):
                # Handle JSON config from dual-stream training
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                # Convert JSON config to expected format
                class ConfigNamespace:
                    def __init__(self, config_dict):
                        model_config = config_dict.get('model_config', {})
                        self.NUM_SAMPLES = model_config.get('num_samples', 7)
                        self.PRED_TYPE = model_config.get('pred_type', 'parameter')
                        self.NUM_PRED = model_config.get('num_pred', 1)
                        self.SAMPLE_RANGE = model_config.get('sample_range', 15)

                parameters = ConfigNamespace(config_data)
                break
            else:
                # Handle standard config format
                parameters, _ = load_config(config_path)
                break

    if parameters is None:
        raise FileNotFoundError(f"No config file found in {model_path}")

    prediction = DualStreamPrediction(parameters, model_name, data_path_calib, model_path, device)
    
    # Generate 4 DDFs for the scan
    pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF = prediction.generate_prediction_DDF(frames, landmark)

    return pred_global_allpts_DDF, pred_global_landmark_DDF, pred_local_allpts_DDF, pred_local_landmark_DDF
