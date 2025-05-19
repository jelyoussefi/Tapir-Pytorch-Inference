import argparse
import time
import torch
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning, 
                        message="torch.tensor results are registered as constants in the trace")
import openvino as ov
from tapnet.tapir_inference import TapirPredictor, TapirPointEncoder, build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
	parser = argparse.ArgumentParser(description="Tapir ONNX Export")
	parser.add_argument("--model", default="models/causal_bootstapir_checkpoint.pt", type=str,
						help="path to Tapir checkpoint")
	parser.add_argument("--resolution", default=480, type=int, help="Input resolution")
	parser.add_argument("--num_points", default=100, type=int, help="Number of points")
	parser.add_argument("--dynamic", action="store_true", help="Use dynamic number of points")
	parser.add_argument("--num_iters", default=4, type=int, help="Number of iterations, 1 for faster inference, 4 for better results")
	parser.add_argument("--output_dir", default="./", type=str, help="Output ONNX file")
	return parser


if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()

	model_path = args.model
	resolution = args.resolution
	num_points = args.num_points
	dynamic = args.dynamic
	num_iters = args.num_iters
	output_dir = args.output_dir

	model = build_model(model_path,(resolution, resolution), num_iters, True, device).eval()
	predictor = TapirPredictor(model).to(device).eval()
	encoder = TapirPointEncoder(model).to(device).eval()

	causal_state_shape = (num_iters, model.num_mixer_blocks, num_points, 2, 512 + 2048)
	causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
	feature_grid = torch.zeros((1, resolution//8, resolution//8, 256), dtype=torch.float32, device=device)
	hires_feats_grid = torch.zeros((1, resolution//4, resolution//4, 128), dtype=torch.float32, device=device)
	query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
	input_frame = torch.zeros((1, 3, resolution, resolution), dtype=torch.float32, device=device)

	# Test model
	query_feats, hires_query_feats = encoder(query_points[None], feature_grid, hires_feats_grid)
	tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

	#ov_encoder = ov.convert_model(encoder, example_input=(query_points[None], feature_grid, hires_feats_grid))
	#ov_encoder.reshape({"query_points": [1,num_points,2], 
	#					"feature_grid": [1,60,60,256], 
	#					"hires_feats_grid": [1,120,120,128]})

	ov_predictor = ov.convert_model(predictor, 
									example_input=(input_frame, query_feats, hires_query_feats, causal_state),
									input=[("frame",[1,3,resolution,resolution]), ("query_feats",[1,num_points,256]), 
									       ("hires_query_feats",[1,num_points,128]),("causal_context",[num_iters,12,num_points,2,2560])])

	# Name predictor outputs using sets
	for i, output in enumerate(ov_predictor.outputs):
		if i == 0:
			output.get_tensor().set_names({"tracks"})
		elif i == 1:
			output.get_tensor().set_names({"visibles"})
		elif i == 2:
			output.get_tensor().set_names({"causal_state"})
		elif i == 3:
			output.get_tensor().set_names({"feature_grid"})
		elif i == 4:
			output.get_tensor().set_names({"hires_feats_grid"})

	print(f"Saving models to {args.output_dir}")
	#ov.save_model(ov_encoder, f"{args.output_dir}/tapir_encoder.xml")
	ov.save_model(ov_predictor, f"{args.output_dir}/tapir.xml")
