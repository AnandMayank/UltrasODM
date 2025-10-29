# this is the function for generating DDF from network predictions
import os
from utils.Prediction import Prediction
from utils.funs import load_config

def predict_ddfs(frames,landmark,data_path_calib,device):
    """
    Args:
        frames (numpy.ndarray): shape=(N, 480, 640),frames in the scan, where N is the number of frames in this scan
        landmark (numpy.ndarray): shape=(100,3), denoting the locations of landmarks. For example, a landmark with location of (10,200,100) denotes the landmark on the 10th frame, with coordinate of (200,100). 
                                  Please note that the first dimension starts from 0, and the second and third dimensions start from 1, which is designed to be consistent with the calibration process.
        data_path_calib (str): path to calibration matrix
        device (str): device to use for prediction, either 'cuda' or 'cpu'

    Returns:
        pred_global_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), global DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame).
                                                The coordinates defination of the 307200 points can be found in the function "reference_image_points" in utils/plot_functions.py
        pred_global_landmark_DDF (numpy.ndarray): shape=(3, 100), global DDF for landmark  
        pred_local_allpts_DDF (numpy.ndarray): shape=(N-1, 3, 307200), local DDF for all pixels, where N-1 is the number of frames in that scan (excluding the first frame) 
        pred_local_landmark_DDF (numpy.ndarray): shape=(3, 100), local DDF for landmark
    
    """
    
    # path to the trained DCL-Net model
    model_path = os.getcwd() + '/results/train_1_test_1'
    model_name = 'saved_model/best_validation_dist_model'
    # parameters used for training the DCL-Net model
    parameters,_ = load_config(model_path+'/'+'test_config.txt')

    prediction = Prediction(parameters,model_name,data_path_calib,model_path,device)
    # generate 4 DDFs for the scan
    pred_global_allpts_DDF,\
    pred_global_landmark_DDF,\
    pred_local_allpts_DDF,\
    pred_local_landmark_DDF = prediction.generate_prediction_DDF(frames,landmark)

    return pred_global_allpts_DDF,pred_global_landmark_DDF,pred_local_allpts_DDF,pred_local_landmark_DDF

