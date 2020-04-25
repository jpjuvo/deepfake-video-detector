import sys
import json
import os

def main():
    detector_dir = "./detector/"
    config_file = "./detector_config.json"

    try:
        if not os.path.isdir(detector_dir):
            raise Exception("detector not found","place detector in the same folder with predict.py")

        if not os.path.isfile(config_file):
            raise Exception("detector_config.json not found","place it in the same folder with predict.py")

        # check if there's no videopath given
        if len(sys.argv) < 2:
            raise Exception("Video not specified","Pass video file path as the first argument")
    
        video_path = sys.argv[1]
        if not os.path.isfile(video_path):
            raise Exception("Video file not found", "{0} is not a file".format(video_path))

        # append detector path to sys path
        sys.path.append(detector_dir)
        from DeepFakeDetector import DeepFakeDetector

        # read configurations
        try:
            with open(config_file) as data:
                config = json.load(data)
                n_first_frames = config['n_first_frames']
                n_spaced_frames = config['n_spaced_frames']
                pretrained_paths = config['pretrained_paths']
                models_root_dir = config['models_root_dir']
        except:
            raise Exception("invalid configuration file",
                            "one or more of these fields are missing: [n_first_frames, n_spaced_frames, pretrained_paths, models_root_dir]")

        try:
            deepFakeDetector = DeepFakeDetector(deepfake_models_directory=models_root_dir,
                                                third_party_models_directory=pretrained_paths,
                                                n_first_frames=n_first_frames,
                                                n_spaced_frames=n_spaced_frames,
                                                verbose=0)
        except:
            raise Exception("detector could not be initialized","check configurations, model files and that all dependencies are installed")
        
        try:
            prediction = deepFakeDetector.Predict(video_path, handleErrors=False)
            print("Prediction: {0}".format(prediction))
            return prediction
        except Exception as inst:
            Exception("detector error", "could not predict the video file")
    except Exception as inst:
        print("{0}:\n{1}".format(*inst.args))
        return -1

main()