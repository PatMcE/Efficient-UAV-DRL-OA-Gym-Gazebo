#! /usr/bin/env python3

from sac_networks import ActorNetwork  # Import ActorNetwork from your SAC networks file
import torch
import onnx

if __name__ == '__main__':
    # Set parameters specific to the Actor network and action space
    action_size = 2  # Action space dimension based on your custom gym environment
    input_dims = (3, 80, 106) # Observation space dimensions: (channels, height, width)
    device = torch.device('cpu')  # Set to CPU for ONNX export

    # Ensure hidden_size matches the trained model's configuration
    hidden_size = 256  # Hidden size used in training
    root_path = '/home/pm/catkin_ws/src/mavros-px4-vehicle/models/'
    load_file = root_path + 'actor.pth'  # Path to the trained model weights
    onnx_file = root_path + 'sac.onnx'  # Path to save the ONNX model

    # Initialize and load the Actor model
    actor = ActorNetwork(input_dims=input_dims, action_size=action_size, device=device, hidden_size=hidden_size)
    actor.load_state_dict(torch.load(load_file, map_location=device))
    
    # Set the model to evaluation mode and move to CPU
    actor = actor.to(device)
    actor.eval()

    # Create example input data matching observation space dimensions
    input_data = torch.ones((1, *input_dims), device=device)

    # Export the Actor model to ONNX
    print("Exporting trained Actor model to ONNX")
    torch.onnx.export(
        actor,                   # The model
        input_data,              # Example input
        onnx_file,               # ONNX file path
        export_params=True,      # Store trained parameter weights
        opset_version=11,        # ONNX opset version (11 or higher recommended)
        do_constant_folding=True # Optimize constant expressions
    )

    ################## Print ONNX Model Information ##################
    # Load and verify the exported ONNX model
    print("Loading and verifying the ONNX model")
    onnx_model = onnx.load(onnx_file)

    # Print layer and weight information
    print("Layers in ONNX model:")
    for layer in onnx_model.graph.node:
        print(f"Layer Name: {layer.name}, Layer Type: {layer.op_type}")
        for input_name in layer.input:
            print(f"\tInput: {input_name}")
        for output_name in layer.output:
            print(f"\tOutput: {output_name}")

    print("\nWeights in ONNX model:")
    for weight in onnx_model.graph.initializer:
        print(f"Weight Name: {weight.name}, Data Type: {onnx.TensorProto.DataType.Name(weight.data_type)}")
    ###############################################################

